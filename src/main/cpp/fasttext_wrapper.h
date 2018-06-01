#ifndef FASTTEXT_WRAPPER_H
#define FASTTEXT_WRAPPER_H

#include "fastText/src/fasttext.h"

/**
 * FastText's wrapper
 */
namespace FastTextWrapper {
    class FastTextApi {
    private:
        fasttext::FastText fastText;
    public:
        FastTextApi();
        // We don't make runCmd() a static method so that Loader.load() is always be called in FastTextApi().
        void runCmd(int, char **);
        bool checkModel(const std::string&);
        void loadModel(const std::string&);
        // Model unloading is not available in fastText C++. This method is added since
        // Java's GC doesn't collect memory allocated by native function calls.
        void unloadModel();
        void test(const std::string&, int32_t);
        // Check if model was loaded successfully
        bool isModelLoaded();
        std::vector<std::string> predict(const std::string&, int32_t);
        std::vector<std::string> predict(const std::string&, int32_t, fasttext::real);
        std::vector<std::pair<fasttext::real,std::string>> predictProba(const std::string&, int32_t);
        std::vector<std::pair<fasttext::real,std::string>> predictProba(const std::string&, int32_t, fasttext::real);
        std::vector<fasttext::real> getWordVector(const std::string&);
        std::vector<fasttext::real> getSentenceVector(const std::string& word);
        std::vector<fasttext::real> getVector(const std::string&);
        std::vector<fasttext::real> getSubwordVector(const std::string&);
        std::vector<std::string> getWords();
        std::vector<std::string> getLabels();
        std::string getWord(int32_t);
        int32_t getNWords();
        std::string getLabel(int32_t);
        int32_t getNLabels();
        double getLr();
        int getLrUpdateRate();
        int getDim();
        int getContextWindowSize();
        int getEpoch();
        int getMinCount();
        int getMinCountLabel();
        int getNSampledNegatives();
        int getWordNgrams();
        std::string getLossName();
        std::string getModelName();
        int getBucket();
        int getMinn();
        int getMaxn();
        double getSamplingThreshold();
        std::string getLabelPrefix();
        std::string getPretrainedVectorsFileName();
    };
}

#endif
