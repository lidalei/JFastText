#include <iostream>
#include <sstream>

#include "fasttext_wrapper.h"
#include "fastText/src/main.cc"

/**
 * FastText's wrapper
 */
namespace FastTextWrapper {
    FastTextApi::FastTextApi() {}

    void FastTextApi::runCmd(int argc, char **argv) {
        main(argc, argv);  // call fastText's main()
    }

    bool FastTextApi::checkModel(const std::string& filename) {
        // Replicate the logic in FastText::checkModel() since it's a private function
        std::ifstream in(filename, std::ifstream::binary);
        int32_t magic;
        int32_t version;
        in.read((char*)&(magic), sizeof(int32_t));
        if (magic != FASTTEXT_FILEFORMAT_MAGIC_INT32) {
            return false;
        }
        in.read((char*)&(version), sizeof(int32_t));
        if (version != FASTTEXT_VERSION) {
            return false;
        }
        return true;
    }

    void FastTextApi::loadModel(const std::string& filename) {
        fastText.loadModel(filename);
    }

    void FastTextApi::unloadModel() {
        // Call shared_ptr's reset() to release memory's ownership so
        // that the memory can be collected by GC (new in C++ 11).
        // Ref: http://en.cppreference.com/w/cpp/memory/shared_ptr/reset
        // Cannot get the shared pointer to args without changing fastText source code.
        // fastText.getArgs().reset();
        fastText.getDictionary().reset();
        fastText.getInputMatrix().reset();
        fastText.getOutputMatrix().reset();
        fastText.getModel().reset();
    }

    void FastTextApi::test(const std::string& filename, int32_t k) {
        std::ifstream ifs(filename);
        if(!ifs.is_open()) {
            std::cerr << "fasttest_wrapper.cc: Test file cannot be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        fastText.test(ifs, k);
    }

    bool FastTextApi::isModelLoaded() {
        return bool(fastText.getModel());
    }

    std::vector<std::string> FastTextApi::predict(const std::string& text, int32_t k) {
        return predict(text, k, 0.0);
    }

    std::vector<std::string> FastTextApi::predict(const std::string& text, int32_t k,
            fasttext::real threshold) {
        std::vector<std::pair<fasttext::real,std::string>> predictions = predictProba(
            text, k, threshold);
        std::vector<std::string> labels;
        for (auto it = predictions.cbegin(); it != predictions.cend(); ++it) {
            labels.push_back(it->second);
        }
        return labels;
    }

    std::vector<std::pair<fasttext::real,std::string>> FastTextApi::predictProba(
            const std::string& text, int32_t k) {
        return predictProba(text, k, 0.0);
    }

    std::vector<std::pair<fasttext::real,std::string>> FastTextApi::predictProba(
            const std::string& text, int32_t k, fasttext::real threshold) {
        std::vector<std::pair<fasttext::real,std::string>> predictions;
        std::istringstream in(text);
        fastText.predict(in, k, predictions, threshold);
        return predictions;
    }

    std::vector<fasttext::real> FastTextApi::getWordVector(const std::string& word) {
        fasttext::Vector vec(getDim());
        fastText.getWordVector(vec, word);
        return std::vector<fasttext::real>(vec.data(), vec.data() + vec.size());
    }

    std::vector<fasttext::real> FastTextApi::getSentenceVector(const std::string& text) {
        fasttext::Vector vec(getDim());
        std::istringstream textstream(text);
        fastText.getSentenceVector(textstream, vec);
        return std::vector<fasttext::real>(vec.data(), vec.data() + vec.size());
    }

    std::vector<fasttext::real> FastTextApi::getVector(const std::string& word) {
        return getWordVector(word);
    }

    std::vector<fasttext::real> FastTextApi::getSubwordVector(const std::string& subword) {
        fasttext::Vector vec(getDim());
        fastText.getSubwordVector(vec, subword);
        return std::vector<fasttext::real>(vec.data(), vec.data() + vec.size());
    }

    std::vector<std::string> FastTextApi::getWords() {
        std::vector<std::string> words;
        int32_t nwords = getNWords();
        for (int32_t i = 0; i < nwords; i++) {
            words.push_back(getWord(i));
        }
        return words;
    }

    std::vector<std::string> FastTextApi::getLabels() {
        std::vector<std::string> labels;
        int32_t nlabels = getNLabels();
        for (int32_t i = 0; i < nlabels; i++) {
            labels.push_back(getLabel(i));
        }
        return labels;
    }

    int32_t FastTextApi::getNWords() {
        return fastText.getDictionary()->nwords();
    }

    std::string FastTextApi::getWord(int32_t i) {
        return fastText.getDictionary()->getWord(i);
    }

    int32_t FastTextApi::getNLabels() {
        return fastText.getDictionary()->nlabels();
    }

    std::string FastTextApi::getLabel(int32_t i) {
        return fastText.getDictionary()->getLabel(i);
    }

    double FastTextApi::getLr() {
        return fastText.getArgs().lr;
    }

    int FastTextApi::getLrUpdateRate() {
        return fastText.getArgs().lrUpdateRate;
    }

    int FastTextApi::getDim() {
        return fastText.getDimension();
    }

    int FastTextApi::getContextWindowSize() {
        return fastText.getArgs().ws;
    }

    int FastTextApi::getEpoch() {
        return fastText.getArgs().epoch;
    }

    int FastTextApi::getMinCount() {
        return fastText.getArgs().minCount;
    }

    int FastTextApi::getMinCountLabel() {
        return fastText.getArgs().minCountLabel;
    }

    int FastTextApi::getNSampledNegatives() {
        return fastText.getArgs().neg;
    }

    int FastTextApi::getWordNgrams() {
        return fastText.getArgs().wordNgrams;
    }

    std::string FastTextApi::getLossName() {
        loss_name lossName = fastText.getArgs().loss;
        if (lossName == loss_name::ns) {
            return "ns";
        } else if (lossName == loss_name::hs) {
            return "hs";
        } else if (lossName == loss_name::softmax) {
            return "softmax";
        } else {
            std::cerr << "fasttest_wrapper.cc: Unrecognized loss name!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    std::string FastTextApi::getModelName() {
        model_name modelName = fastText.getArgs().model;
        if (modelName == model_name::cbow) {
            return "cbow";
        } else if (modelName == model_name::sg) {
            return "sg";
        } else if (modelName == model_name::sup) {
            return "sup";
        } else {
            std::cerr << "fasttest_wrapper.cc: Unrecognized model name!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    int FastTextApi::getBucket() {
        return fastText.getArgs().bucket;
    }

    int FastTextApi::getMinn() {
        return fastText.getArgs().minn;
    }

    int FastTextApi::getMaxn() {
        return fastText.getArgs().maxn;
    }

    double FastTextApi::getSamplingThreshold() {
        return fastText.getArgs().t;
    }

    std::string FastTextApi::getLabelPrefix() {
        return fastText.getArgs().label;
    }

    std::string FastTextApi::getPretrainedVectorsFileName() {
        return fastText.getArgs().pretrainedVectors;
    }
}
