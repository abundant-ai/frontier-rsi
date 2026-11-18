def plan(now, ready, busy, free_bytes, bytes_per_token):
    actions = []
    selected = []
    for gpu in range(2):
        if not busy[gpu]:
            for row in ready:
                if row[0] not in selected:
                    quantum = row[3] - row[2]
                    steps = quantum - 1
                    required = max(row[5], (row[1] + row[2] + steps - 1) * bytes_per_token if steps else 0)
                    if required - row[5] <= free_bytes:
                        actions.append((gpu, (row[0],), quantum))
                        selected.append(row[0])
                        free_bytes -= required - row[5]
                        break
    return actions
