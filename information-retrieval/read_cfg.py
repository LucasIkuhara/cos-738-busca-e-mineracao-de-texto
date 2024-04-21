def read_cfg(filename: str) -> dict[str, list[str]]:
    with open(filename, "r") as f:

        cfg = {}

        cmds = f.readlines()
        for raw_line in cmds:
            command, arg = raw_line.split("=")
            arg = arg.replace("\n", "")

            if command in cfg:
                cfg[command].append(arg)

            else:
                cfg[command] = [arg]
        
        return cfg


if __name__ == "__main__":
    cfg = read_cfg("PC.CFG")
    print(cfg)
