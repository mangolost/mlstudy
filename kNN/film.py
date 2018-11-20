class Film:
    id = None
    name = None
    type = None
    fight_num = None
    kiss_num = None

    def __init__(self, id=None, name=None, type=None, fight_num=None, kiss_num=None):
        self.id = id
        self.name = name
        self.type = type
        self.fight_num = fight_num
        self.kiss_num = kiss_num

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, item, value):
        self.__dict__[item] = value
