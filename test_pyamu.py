import pyamu as amu
import pytest


@pytest.fixture(scope="class")
def frLemmatizeResult():
    return amu.frLemmatize("nous courons et dansons toute la joourné")

class TestNaturalLanguageProcessing:
    def test_argument_type(self):
        with pytest.raises(TypeError):
            amu.frLemmatize(123)
        with pytest.raises(TypeError):
            amu.frLemmatize([])
        with pytest.raises(TypeError):
            amu.frLemmatize(None)

    def test_return_type(self, frLemmatizeResult):
        assert isinstance(frLemmatizeResult, list)

    def test_items_num(self, frLemmatizeResult):
        data = "nous courons et dansons toute la joourné".split(' ')
        assert len(frLemmatizeResult) == len(data)

    def test_attribute_word(self, frLemmatizeResult):
        data = "nous courons et dansons toute la joourné".split(' ')
        for result in frLemmatizeResult:
            assert result.word in data

    def test_attribute_lemma(self, frLemmatizeResult):
        data = ["nous", "courir", "et", "danson", "tout", "le", "joourné"]
        for result in frLemmatizeResult:
            assert result.lemma in data

    def test_attribute_analogy(self, frLemmatizeResult):
        data = [None, "Bétourné"]
        for result in frLemmatizeResult:
            assert result.analogy in data
         
    def test_attribute_analogyLemma(self, frLemmatizeResult):
        data = [None, "Bétourné"]
        for result in frLemmatizeResult:
            assert result.analogyLemma in data

    def test_attribute_corrected(self, frLemmatizeResult):
        data = "nous courons et dansons toute la journée".split(' ')
        for result in frLemmatizeResult:
            assert result.corrected in data
    
    def test_attribute_correctedLemma(self, frLemmatizeResult):
        data = ["nous", "courir", "et", "danson", "tout", "le", "journée"]
        for result in frLemmatizeResult:
            assert result.correctedLemma in data

    def test_attribute_similarity(self, frLemmatizeResult):
        for result in frLemmatizeResult:
            assert isinstance(result.similarity, float)
            assert result.similarity >= 0
            assert result.similarity <= 1

    def test_attribute_probability(self, frLemmatizeResult):
        for result in frLemmatizeResult:
            assert isinstance(result.probability, float)
            assert result.probability >= 0
            assert result.probability <= 1


class TestCombinatorialAlgorithms:
    def test_argument_type(self):
        result = amu.lexicalOrder(3, 2)
        assert result == [
            [1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2],
            [2, 1, 1], [2, 1, 2], [2, 2, 1], [2, 2, 2]
        ]

    def test_bijectionSubsets(self):
        result = amu.bijectionSubsets(2)
        assert result == {(0, 0): [], (1, 0): [1], (1, 1): [1, 2], (0, 1): [2]}

    def test_lexicalPosition(self):
        result = amu.lexicalPosition(5, [2, 3, 5])
        assert result == 13
    
    def test_lexicalSubsetFromPosition(self):
        result = amu.lexicalSubsetFromPosition(5, 13)
        assert result == [2, 3, 5]

    def test_lexicalOrderRecursive(self):
        result = amu.lexicalOrderRecursive(2, 3)
        assert result == [
            [1, 1], [1, 2], [1, 3], [2, 1], [2, 2],
            [2, 3], [3, 1], [3, 2], [3, 3]
        ]

    def test_grayOrder(self):
        result = amu.grayOrder(2)
        assert result == [[], [2], [1, 2], [1]]

    def test_grayRank(self):
        result = amu.grayRank(4, [1, 3])
        assert result == 12

    def test_graySubsetFromRank(self):
        result = amu.graySubsetFromRank(4, 12)
        assert result == [1, 3]
    
    def test_kSubsetSuccessor(self):
        result = amu.kSubsetSuccessor(8, 4, [1, 2, 7, 8])
        assert result == [1, 3, 4, 5]

    def test_kSubsetRank(self):
        result = amu.kSubsetRank(5, 3, [2, 3, 5])
        assert result == 7

    def test_kSubsetFromRank(self):
        result = amu.kSubsetFromRank(5, 3, 7)
        assert result == [2, 3, 5]

    def test_subsetDivision(self):
        result = amu.subsetDivision(3)
        assert result == [
            [ [1, 2, 3] ], 
            [ [1, 2], [3] ], 
            [ [1], [2], [3] ],
            [ [1], [2, 3] ],
            [ [1, 3], [2] ]
        ]

    def test_rgfFunction(self):
        result = amu.rgfFunction(10, [[3, 6, 7], [1, 2], [5, 8, 9], [4, 10]])
        assert result == [1, 1, 2, 3, 4, 2, 2, 4, 4, 3]
    
    def test_rgfGenerate(self):
        result = amu.rgfGenerate(3)
        assert result == [
            [1, 1, 1], [1, 1, 2], [1, 2, 1], [1, 2, 2], [1, 2, 3]
        ]

    def test_permutationSuccessor(self):
        result = amu.permutationSuccessor([3, 6, 2, 7, 5, 4, 1])
        assert result == [3, 6, 4, 1, 2, 5, 7]

    def test_permutationRank(self):
        result = amu.permutationRank([2, 4, 1, 3])
        assert result == 10

    def test_permutationFromRank(self):
        result = amu.permutationFromRank(4, 10)
        assert result == [2, 4, 1, 3]

    def test_numDivision(self):
        result = amu.numDivision(7, 3)
        assert result == 4

    def test_conjugateDivision(self):
        result = amu.conjugateDivision([4, 3, 2, 2, 1])
        assert result == [5, 4, 2, 1]

    def test_allDivisions(self):
        result = amu.allDivisions(5)
        assert result == [
            [1, 1, 1, 1, 1], [2, 1, 1, 1], [2, 2, 1], 
            [3, 1, 1], [3, 2], [4, 1], [5]
        ]

    def test_prufer(self):
        result = amu.prufer(8, [{1, 7}, {1, 6}, {1, 4}, {6, 8}, {4, 5}, {8, 3}, {8, 2}])
        assert result == [1, 4, 1, 8, 8, 6]

    def test_fromPrufer(self):
        result = amu.fromPrufer([1, 4, 1, 8, 8, 6])
        assert result == [
            {1, 7}, {4, 5}, {1, 4}, {8, 3}, {8, 2}, {8, 6}, {1, 6}
        ]

    def test_pruferRank(self):
        result = amu.pruferRank([1, 4, 1, 8, 8, 6])
        assert result == 12797

    def test_pruferFromRank(self):
        result = amu.pruferFromRank(8, 12797)
        assert result == [1, 4, 1, 8, 8, 6]

    def test_prim(self):
        V = [1, 2, 3, 4, 5, 6, 7]
        E = [{1,2}, {1,3}, {2,3}, {2,5}, {3,6}, {2,4}, {3,4}, {4,5}, {4,6}, {5,6}, {5,7}, {6,7}]
        w = {(1,2):1, (1,3):4, (2,3):2, (2,5):5, (3,6):2, (2,4):3, (3,4):3, (4,5):1, (4,6):4, (5,6):3, (5,7):5, (6,7):4}
        g = amu.G(V, E, w)
        result = amu.prim(g)
        assert result == {
            'weight': 13,
            'E': [{1, 2}, {2, 3}, {3, 6}, {2, 4}, {4, 5}, {6, 7}]
        }

class TestNeuralNetwork:
    def test_displayGates(self, capfd):
        amu.displayGates()
        out, err = capfd.readouterr()
        assert "Gate Not\nu1 = 0, result = 1\nu1 = 1, result = 0" in out
    
    def test_displayGradientMethod(self, capfd):
        amu.displayGradientMethod()
        out, err = capfd.readouterr()
        assert "Global min" in out
        assert "Local min" in out
    
    def test_displayBackPropagation(self, capfd):
        amu.displayBackPropagation()
        out, err = capfd.readouterr()
        assert "y = " in out

        s_old, w_old = [0, 1, 2], [[0, 1, 2], [0, 1, 2]]
        s_new, w_new, y = amu.displayBackPropagation.update(s_old, w_old)
        assert round(sum(y)) == 4
    
    def test_displayBoltzmannMachine(self, capfd):
        max_t = 10
        amu.displayBoltzmannMachine(max_t)
        out, err = capfd.readouterr()
        assert out.count("[") == max_t * 5 + 2 * 5
