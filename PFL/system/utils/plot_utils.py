

class color_2:
    def __init__(self):
        super().__init__()
        self.pink = self.hex_to_rgb('E8D7CD')
        self.yellow = self.hex_to_rgb('EBC691')
        self.deep_pink = self.hex_to_rgb('CD968E')
        self.red = self.hex_to_rgb('711E24')
        self.grey = self.hex_to_rgb('A8B1CA')
        self.lite_blue = self.hex_to_rgb('969FC1')
        self.blue = self.hex_to_rgb('CD968E')
        self.deep_blue = self.hex_to_rgb('4251A3')

        self.color_list = [ self.red,self.pink, self.yellow, self.deep_pink, self.grey, self.lite_blue, self.blue, self.deep_blue]
    def hex_to_rgb(self, hex_string:str):
        # Remove the '#' if present
        hex_string = hex_string.lstrip('#')

        # Convert the hex string to RGB values
        rgb = tuple(int(hex_string[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

        return rgb



