def write_tsv(segments, path, tier="gestures", label="gesture"):
    with open(path, "w", encoding="utf-8") as f:
        f.write("tier\tstart_ms\tend_ms\tvalue\n")
        for s, e in segments:
            f.write(f"{tier}\t{s}\t{e}\t{label}\n")
