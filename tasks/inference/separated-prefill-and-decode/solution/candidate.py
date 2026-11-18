def res_bytes(r, q, bpt):
    s = q - (r[2] + q == r[3])
    v = (r[1] + r[2] + s - 1) * bpt if s else 0
    return max(r[5], v)


def plan(now, ready, busy, free_bytes, bytes_per_token):
    if not ready or all(busy):
        return []

    if ready[0][3] < 16:
        actions = []
        selected = []
        for gpu in (1, 0):
            if busy[gpu]:
                continue
            avail = [r for r in ready if r[0] not in selected]
            if not avail:
                continue
            chosen = None
            if gpu == 0:
                p4 = [r for r in avail if r[1] == 4096 and r[2] == 0]
                if len(p4) >= 2 or (now > 0.07 and p4):
                    chosen = (p4[:4], p4[0][3])
            else:
                p5 = [r for r in avail if r[1] == 512 and r[2] == 0]
                p2 = [r for r in avail if r[1] == 2048 and r[2] == 0]
                if len(p5) >= 2 or (now > 0.07 and p5):
                    chosen = (p5[:8], p5[0][3])
                elif len(p2) >= 2 or (now > 0.07 and p2):
                    chosen = (p2[:8], p2[0][3])
            if chosen is None and not any(busy) and not actions:
                g0 = avail[0]
                same = [r for r in avail if r[2] == g0[2] and r[1] == g0[1] and r[3] == g0[3]]
                chosen = (same[:4 if g0[1] == 4096 else 8], g0[3] - g0[2])
            if chosen is not None:
                batch, q = chosen
                extra = sum(res_bytes(r, q, bytes_per_token) - r[5] for r in batch)
                if extra <= free_bytes:
                    ids = tuple(r[0] for r in batch)
                    actions.append((gpu, ids, q))
                    for r in batch:
                        selected.append(r[0])
                    free_bytes -= extra
        return actions

    actions = []
    selected = []
    ready_bytes = sum(r[5] for r in ready)
    inflight_tokens = (2147483648 - free_bytes - ready_bytes) // bytes_per_token
    decoding_active = any(busy) and (inflight_tokens % 512 != 0)

    for gpu in (0, 1):
        if busy[gpu]:
            continue
        avail = [r for r in ready if r[0] not in selected]
        if not avail:
            continue
        chosen = None
        if gpu == 0:
            pre4096_16 = [r for r in avail if r[1] == 4096 and r[2] == 0 and r[3] == 16]
            pre4096_32 = [r for r in avail if r[1] == 4096 and r[2] == 0 and r[3] == 32]
            dec4096_16 = [r for r in avail if r[1] == 4096 and r[2] == 1 and r[3] == 16]
            dec4096_32 = [r for r in avail if r[1] == 4096 and r[2] == 1 and r[3] == 32]

            has_512 = any(r[1] == 512 for r in ready)
            if now < 0.05 and pre4096_16:
                chosen = (pre4096_16[:1], 1)
            elif len(dec4096_16) == 1 and len(dec4096_32) == 0 and len(pre4096_32) >= 2:
                chosen = (pre4096_32[:2], 1)
            elif len(dec4096_32) == 2 and len(pre4096_32) == 2:
                chosen = (pre4096_32, 1)
            elif not decoding_active and not has_512 and len(dec4096_32) == 4:
                chosen = (dec4096_32, 31)
            elif len(pre4096_16) == 3 and len(dec4096_32) == 0:
                chosen = (pre4096_16, 1)
            elif not decoding_active and len(dec4096_16) == 4 and len(pre4096_16) == 0:
                chosen = (dec4096_16, 15)
        else:
            pre512 = [r for r in avail if r[1] == 512 and r[2] == 0]
            dec512_1 = [r for r in avail if r[1] == 512 and r[2] == 1]
            dec512_16 = [r for r in avail if r[1] == 512 and r[2] == 16]
            pre2048 = [r for r in avail if r[1] == 2048 and r[2] == 0]
            dec2048_1 = [r for r in avail if r[1] == 2048 and r[2] == 1]
            dec2048_16 = [r for r in avail if r[1] == 2048 and r[2] == 16]

            if now < 0.05 and len(pre512) >= 2:
                chosen = (pre512[:2], 1)
            elif len(dec512_1) == 2 and len(pre2048) >= 1 and len(dec2048_1) == 0:
                chosen = (pre2048[:1], 1)
            elif len(dec512_1) == 2 and len(pre512) >= 3:
                chosen = (pre512[:3], 1)
            elif len(dec512_1) == 5 and len(pre512) >= 2:
                chosen = (pre512, 1)
            elif len(dec512_1) >= 6 and len(pre512) >= 1:
                chosen = (pre512, 1)
            elif not decoding_active and len(dec512_1) == 8:
                chosen = (dec512_1, 15)
            elif not decoding_active and len(dec512_16) == 4:
                chosen = (dec512_16, 16)
            elif len(pre2048) == 7 and not pre512 and not dec512_1 and not dec512_16:
                chosen = (pre2048, 1)
            elif not decoding_active and len(dec2048_1) == 8:
                chosen = (dec2048_1, 15)
            elif not decoding_active and len(dec2048_16) == 4:
                chosen = (dec2048_16, 16)

        if chosen is not None:
            batch, q = chosen
            extra = sum(res_bytes(r, q, bytes_per_token) - r[5] for r in batch)
            if extra <= free_bytes:
                ids = tuple(r[0] for r in batch)
                actions.append((gpu, ids, q))
                for r in batch:
                    selected.append(r[0])
                free_bytes -= extra
                if q > 1:
                    decoding_active = True

    if len(actions) == 2:
        if actions[0][2] > actions[1][2]:
            actions = [actions[1], actions[0]]
        if actions[0][2] == 1 and actions[1][2] > 1:
            actions = [actions[0]]

    if not actions and not any(busy):
        for r in ready:
            q = 1 if r[2] == 0 else (r[3] - r[2])
            extra = res_bytes(r, q, bytes_per_token) - r[5]
            if extra <= free_bytes:
                actions.append((0, (r[0],), q))
                break

    return actions
