from kNN.categoryService import CategoryService
from kNN.film import Film

categoryService = CategoryService()

film1 = Film()
film1.kiss_num = 22
film1.fight_num = 77
print(categoryService.calType(film1))

film2 = Film()
film2.kiss_num = 60
film2.fight_num = 25
print(categoryService.calType(film2))


