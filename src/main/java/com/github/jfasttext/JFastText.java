package com.github.jfasttext;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import org.bytedeco.javacpp.PointerPointer;

public class JFastText {

  private FastTextWrapper.FastTextApi fta;

  public JFastText() {
    fta = new FastTextWrapper.FastTextApi();
  }

  public void runCmd(String[] args) {
    // Prepend "fasttext" to the argument list so that it is compatible with C++'s main()
    String[] cArgs = new String[args.length + 1];
    cArgs[0] = "fasttext";
    System.arraycopy(args, 0, cArgs, 1, args.length);
    fta.runCmd(cArgs.length, new PointerPointer(cArgs));
  }

  public void loadModel(String modelFile)
      throws FileNotFoundException, IllegalArgumentException, ExceptionInInitializerError {
    if (!new File(modelFile).exists()) {
      throw new FileNotFoundException("Model file doesn't exist!");
    }
    if (!fta.checkModel(modelFile)) {
      throw new IllegalArgumentException(
          "Model file's format is not compatible with this JFastText version!");
    }
    fta.loadModel(modelFile);

    if (!fta.isModelLoaded()) {
      throw new ExceptionInInitializerError(
          "Invalid model format. Check https://github.com/facebookresearch/fastText/issues/332");
    }
  }

  public void loadDefaultLanguageDetectionModel() {
    String pathToDefaultModel = "lid.176.ftz";
    InputStream modelIS = JFastText.class.getResourceAsStream(pathToDefaultModel);

    // create a temp file and copy the content of the default model to it
    Path pathToTempModel;
    try {
      pathToTempModel =
          Files.createTempFile(
              "lid.176.",
              ".ftz",
              PosixFilePermissions.asFileAttribute(
                  new HashSet<>(
                      Arrays.asList(
                          PosixFilePermission.OWNER_READ, PosixFilePermission.OWNER_WRITE))));

      Files.copy(modelIS, pathToTempModel, StandardCopyOption.REPLACE_EXISTING);
    } catch (IOException e) {
      e.printStackTrace();
      return;
    }

    fta.loadModel(pathToTempModel.toString());

    if (!fta.isModelLoaded()) {
      throw new ExceptionInInitializerError(
          "Invalid model format. Check https://github.com/facebookresearch/fastText/issues/332");
    }
  }

  public void unloadModel() {
    fta.unloadModel();
  }

  public void test(String testFile) {
    test(testFile, 1);
  }

  public void test(String testFile, int k) {
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive");
    }
    fta.test(testFile, k);
  }

  public String predict(String text) {
    List<String> predictions = predict(text, 1);
    return predictions.size() > 0 ? predictions.get(0) : "und";
  }

  public List<String> predict(String text, int k) {
    return predict(text, k, 0.0f);
  }

  public List<String> predict(String text, int k, float threshold) {
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive");
    }
    FastTextWrapper.StringVector sv = fta.predict(text, k, threshold);
    List<String> predictions = new ArrayList<>();
    for (int i = 0; i < sv.size(); i++) {
      predictions.add(sv.get(i).getString());
    }
    return predictions;
  }

  public ProbLabel predictProba(String text) {
    List<ProbLabel> probaPredictions = predictProba(text, 1);
    return probaPredictions.size() > 0 ? probaPredictions.get(0) : new ProbLabel(0, "und");
  }

  public List<ProbLabel> predictProba(String text, int k) {
    return predictProba(text, k, 0.0f);
  }

  public List<ProbLabel> predictProba(String text, int k, float threshold) {
    if (k <= 0) {
      throw new IllegalArgumentException("k must be positive");
    }
    FastTextWrapper.FloatStringPairVector fspv = fta.predictProba(text, k, threshold);
    List<ProbLabel> probaPredictions = new ArrayList<>();
    for (int i = 0; i < fspv.size(); i++) {
      float logProb = fspv.first(i);
      String label = fspv.second(i).getString();
      probaPredictions.add(new ProbLabel(logProb, label));
    }
    return probaPredictions;
  }

  public List<Float> getWordVector(String word) {
    FastTextWrapper.RealVector rv = fta.getWordVector(word);
    List<Float> wordVec = new ArrayList<>();
    for (int i = 0; i < rv.size(); i++) {
      wordVec.add(rv.get(i));
    }
    return wordVec;
  }

  public List<Float> getSentenceVector(String sentence) {
    FastTextWrapper.RealVector rv = fta.getSentenceVector(sentence);
    List<Float> sentVec = new ArrayList<>();
    for (int i = 0; i < rv.size(); i++) {
      sentVec.add(rv.get(i));
    }
    return sentVec;
  }

  public List<Float> getSubwordVector(String subword) {
    FastTextWrapper.RealVector rv = fta.getSubwordVector(subword);
    List<Float> subwordVec = new ArrayList<>();
    for (int i = 0; i < rv.size(); i++) {
      subwordVec.add(rv.get(i));
    }
    return subwordVec;
  }

  public int getNWords() {
    return fta.getNWords();
  }

  public List<String> getWords() {
    return stringVec2Strings(fta.getWords());
  }

  public int getNLabels() {
    return fta.getNLabels();
  }

  public List<String> getLabels() {
    return stringVec2Strings(fta.getLabels());
  }

  public double getLr() {
    return fta.getLr();
  }

  public int getLrUpdateRate() {
    return fta.getLrUpdateRate();
  }

  public int getDim() {
    return fta.getDim();
  }

  public int getContextWindowSize() {
    return fta.getContextWindowSize();
  }

  public int getEpoch() {
    return fta.getEpoch();
  }

  public int getMinCount() {
    return fta.getMinCount();
  }

  public int getMinCountLabel() {
    return fta.getMinCountLabel();
  }

  public int getNSampledNegatives() {
    return fta.getNSampledNegatives();
  }

  public int getWordNgrams() {
    return fta.getWordNgrams();
  }

  public String getLossName() {
    return fta.getLossName().getString();
  }

  public String getModelName() {
    return fta.getModelName().getString();
  }

  public int getBucket() {
    return fta.getBucket();
  }

  public int getMinn() {
    return fta.getMinn();
  }

  public int getMaxn() {
    return fta.getMaxn();
  }

  public double getSamplingThreshold() {
    return fta.getSamplingThreshold();
  }

  public String getLabelPrefix() {
    return fta.getLabelPrefix().getString();
  }

  public String getPretrainedVectorsFileName() {
    return fta.getPretrainedVectorsFileName().getString();
  }

  private static List<String> stringVec2Strings(FastTextWrapper.StringVector sv) {
    List<String> strings = new ArrayList<>();
    for (int i = 0; i < sv.size(); i++) {
      strings.add(sv.get(i).getString());
    }
    return strings;
  }

  public static class ProbLabel {
    public float logProb;
    public String label;

    public ProbLabel(float logProb, String label) {
      this.logProb = logProb;
      this.label = label;
    }

    @Override
    public String toString() {
      return String.format("logProb = %f, label = %s", logProb, label);
    }
  }

  public static void main(String[] args) {
    JFastText jft = new JFastText();
    jft.runCmd(args);
    //    jft.loadDefaultLanguageDetectionModel();
    //    jft.unloadModel();
  }
}
