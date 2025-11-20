from __future__ import annotations
import psycopg2
import json
import re
from datetime import datetime
from collections import defaultdict

DB_CONFIG = {
    "dbname": "postgres",
    "user": "BIGDATA",
    "password": "PASSWORD",
    "host": "localhost",
    "port": 5432,
}

ARCH_RULES = {
    "transformer": ["transformer", "gpt", "bert", "t5", "llama", "mistral", "falcon",
                    "roberta", "xlm", "vit", "clip", "blip", "whisper", "wav2vec",
                    "deberta", "bart", "distilbert", "albert", "electra"],
    "diffusion": ["diffusion", "stable-diffusion", "sdxl", "unet", "latent-diffusion",
                  "controlnet", "flux", "kandinsky", "sd3"],
    "cnn": ["cnn", "resnet", "efficientnet", "mobilenet", "densenet", "vgg", "inception"],
    "rnn": ["rnn", "lstm", "gru", "seq2seq"],
    "gan": ["gan", "stylegan", "biggan", "cyclegan", "pix2pix"],
    "vae": ["vae", "autoencoder", "auto-encoder"],
    "gnn": ["gnn", "graphsage", "gcn", "gat"],
    "mlp": ["mlp", "perceptron", "feedforward"],
}
LIB_HINTS = {"diffusers": "diffusion"}

_num_re = re.compile(r"^[+-]?\d+(\.\d+)?$")


def parse_created_at(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        if v >= 1e12:
            return datetime.utcfromtimestamp(v / 1000)
        if v >= 1e9:
            return datetime.utcfromtimestamp(v)
    s = str(x)
    if _num_re.match(s):
        v = float(s)
        if v >= 1e12:
            return datetime.utcfromtimestamp(v / 1000)
        if v >= 1e9:
            return datetime.utcfromtimestamp(v)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def normalize_tags(x):
    if not x:
        return []
    if isinstance(x, list):
        return [str(t).lower() for t in x]
    try:
        parsed = json.loads(x)
        if isinstance(parsed, list):
            return [str(t).lower() for t in parsed]
    except Exception:
        return []
    return []


def infer_arch(model_id, tags, pipeline_tag, library):
    s_all = " ".join([model_id] + tags).lower()
    for arch, keys in ARCH_RULES.items():
        if any(k in s_all for k in keys):
            return arch
    if library and library.lower() in LIB_HINTS:
        return LIB_HINTS[library.lower()]
    return "other"


def main():
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)

    create_sql = """
    DROP TABLE IF EXISTS hf_architecture;
    CREATE TABLE hf_architecture (
        period DATE NOT NULL,
        cnn INTEGER DEFAULT 0,
        diffusion INTEGER DEFAULT 0,
        gan INTEGER DEFAULT 0,
        transformer INTEGER DEFAULT 0,
        rnn INTEGER DEFAULT 0,
        vae INTEGER DEFAULT 0,
        gnn INTEGER DEFAULT 0,
        mlp INTEGER DEFAULT 0,
        other INTEGER DEFAULT 0
    );
    """
    with conn.cursor() as cur:
        cur.execute(create_sql)
        cur.execute("TRUNCATE hf_architecture")
        conn.commit()
    print("Table created and truncated.")

    select_sql = """
        SELECT model_id, createdat, tags, pipeline_tag,
               (raw_data::jsonb->>'library_name') AS library_name
        FROM hf_models
        WHERE createdat IS NOT NULL;
    """
    with conn.cursor() as cur:
        cur.execute(select_sql)
        rows = cur.fetchall()

    agg = defaultdict(lambda: {arch: 0 for arch in list(ARCH_RULES.keys()) + ["other"]})

    for model_id, createdat, tags, pipeline_tag, library_name in rows:
        dt = parse_created_at(createdat)
        if not dt:
            continue
        period = datetime(dt.year, dt.month, 1)
        tag_list = normalize_tags(tags)
        arch = infer_arch(model_id, tag_list, pipeline_tag, library_name)
        agg[period][arch] += 1

    print("Writing aggregated data to DB...")
    with conn.cursor() as cur:
        for period in sorted(agg.keys()):
            counts = agg[period]
            cur.execute(
                """
                INSERT INTO hf_architecture
                (period, cnn, diffusion, gan, transformer, rnn, vae, gnn, mlp, other)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    period,
                    counts.get("cnn", 0),
                    counts.get("diffusion", 0),
                    counts.get("gan", 0),
                    counts.get("transformer", 0),
                    counts.get("rnn", 0),
                    counts.get("vae", 0),
                    counts.get("gnn", 0),
                    counts.get("mlp", 0),
                    counts.get("other", 0),
                )
            )
    conn.commit()
    conn.close()
    print("DONE. Aggregated data loaded.")


if __name__ == "__main__":
    main()
