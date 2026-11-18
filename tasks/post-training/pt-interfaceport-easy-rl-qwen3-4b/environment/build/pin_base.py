"""Pin a model directory's chat template so that every prompt is formatted the same way on every day: each
`strftime_now(...)` (today's date) becomes a fixed date, and where the template has a reasoning mode
(`enable_thinking`), it is switched off. The pinned template is written to chat_template.jinja (and chat_template.json,
if the model has one); tokenizer_config.json's own copy is removed, so there is one template.

  python3 pin_base.py <model_dir> <year> <month> <day>
"""
import datetime, json, os, re, sys


def template_source(model_dir):
    p = os.path.join(model_dir, "chat_template.jinja")
    if os.path.isfile(p):
        return open(p).read()
    cfg = json.load(open(os.path.join(model_dir, "tokenizer_config.json")))
    t = cfg.get("chat_template")
    if isinstance(t, list):                              # [{"name": "default", "template": ...}, ...]
        t = next((x["template"] for x in t if x.get("name") == "default"), t[0]["template"])
    if not t:
        p = os.path.join(model_dir, "chat_template.json")
        t = json.load(open(p))["chat_template"] if os.path.isfile(p) else None
    if not t:
        raise SystemExit(f"{model_dir}: no chat template")
    return t


def pinned(tpl, date):
    d = datetime.date(*date)
    tpl = re.sub(r"strftime_now\(\s*(['\"])(.*?)\1\s*\)", lambda m: json.dumps(d.strftime(m.group(2))), tpl)
    if "enable_thinking" in tpl:
        tpl = "{%- set enable_thinking = false -%}\n" + tpl
    return tpl


def pin(model_dir, date):
    tpl = pinned(template_source(model_dir), date)
    with open(os.path.join(model_dir, "chat_template.jinja"), "w") as f:
        f.write(tpl)
    p = os.path.join(model_dir, "tokenizer_config.json")
    cfg = json.load(open(p))
    if cfg.pop("chat_template", None) is not None:
        json.dump(cfg, open(p, "w"), indent=2, ensure_ascii=False)
    p = os.path.join(model_dir, "chat_template.json")
    if os.path.isfile(p):
        json.dump({"chat_template": tpl}, open(p, "w"), indent=2, ensure_ascii=False)
    return tpl


if __name__ == "__main__":
    pin(sys.argv[1], tuple(int(x) for x in sys.argv[2:5]))
    print("pinned", sys.argv[1])
