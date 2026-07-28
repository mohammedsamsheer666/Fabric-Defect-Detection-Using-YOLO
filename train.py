
import subprocess

darknet_path = "./darknet"   
data_file = "data/obj.data"
cfg_file = "cfg/Custom.cfg"
pretrained_weights = "darknet53.conv.74"


def train():
    command = [
        darknet_path,
        "detector",
        "train",
        data_file,
        cfg_file,
        pretrained_weights,
        "-dont_show",
        "-map"
    ]

    print("Starting training...\n")
    subprocess.run(command)


if __name__ == "__main__":
    train()
