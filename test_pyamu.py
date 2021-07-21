# @pytest.mark.parametrize("i", list(range(len(frLemmatizeSentenceSplit))))
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