package com.github.jfasttext;

import com.github.jfasttext.JFastText.ProbLabel;
import java.util.List;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public class JFastTextTest {

  @Test
  public void test01TrainSupervisedCmd() {
    System.out.println("Training supervised model ...");
    JFastText jft = new JFastText();
    jft.runCmd(
        new String[] {
          "supervised",
          "-input",
          "src/test/resources/data/labeled_data.txt",
          "-output",
          "src/test/resources/models/supervised.model"
        });
  }

  @Test
  public void test02TrainSkipgramCmd() {
    System.out.println("Training skipgram word-embedding ...");
    JFastText jft = new JFastText();
    jft.runCmd(
        new String[] {
          "skipgram",
          "-input",
          "src/test/resources/data/unlabeled_data.txt",
          "-output",
          "src/test/resources/models/skipgram.model",
          "-bucket",
          "100",
          "-minCount",
          "1"
        });
  }

  @Test
  public void test03TrainCbowCmd() {
    System.out.println("Training cbow word-embedding ...");
    JFastText jft = new JFastText();
    jft.runCmd(
        new String[] {
          "skipgram",
          "-input",
          "src/test/resources/data/unlabeled_data.txt",
          "-output",
          "src/test/resources/models/cbow.model",
          "-bucket",
          "100",
          "-minCount",
          "1"
        });
  }

  @Test
  public void test04Predict() throws Exception {
    JFastText jft = new JFastText();
    jft.loadModel("src/test/resources/models/supervised.model.bin");
    String text = "I like soccer";
    String predictedLabel = jft.predict(text);
    System.out.printf("Text: '%s', label: '%s'\n", text, predictedLabel);
  }

  @Test
  public void test05PredictProba() throws Exception {
    JFastText jft = new JFastText();
    jft.loadModel("src/test/resources/models/supervised.model.bin");
    String text = "What is the most popular sport in the US ?";
    JFastText.ProbLabel predictedProbLabel = jft.predictProba(text);
    System.out.printf(
        "Text: '%s', label: '%s', probability: %f\n",
        text, predictedProbLabel.label, Math.exp(predictedProbLabel.logProb));
  }

  @Test
  public void test06MultiPredictProba() throws Exception {
    JFastText jft = new JFastText();
    jft.loadModel("src/test/resources/models/supervised.model.bin");
    String text = "Do you like soccer ?";
    System.out.printf("Text: '%s'\n", text);
    for (JFastText.ProbLabel predictedProbLabel : jft.predictProba(text, 2)) {
      System.out.printf(
          "\tlabel: '%s', probability: %f\n",
          predictedProbLabel.label, Math.exp(predictedProbLabel.logProb));
    }
  }

  @Test
  public void test07GetWordVector() throws Exception {
    JFastText jft = new JFastText();
    jft.loadModel("src/test/resources/models/supervised.model.bin");
    String word = "soccer";
    List<Float> vec = jft.getWordVector(word);
    System.out.printf("Word embedding vector of '%s': %s\n", word, vec);
  }

  /** Test retrieving model's information: words, labels, learning rate, etc. */
  @Test
  public void test08ModelInfo() throws Exception {
    System.out.println("Supervised model information:");
    JFastText jft = new JFastText();
    jft.loadModel("src/test/resources/models/supervised.model.bin");
    System.out.printf("\tnumber of words = %d\n", jft.getNWords());
    System.out.printf("\twords = %s\n", jft.getWords());
    System.out.printf("\tlearning rate = %g\n", jft.getLr());
    System.out.printf("\tdimension = %d\n", jft.getDim());
    System.out.printf("\tcontext window size = %d\n", jft.getContextWindowSize());
    System.out.printf("\tepoch = %d\n", jft.getEpoch());
    System.out.printf("\tnumber of sampled negatives = %d\n", jft.getNSampledNegatives());
    System.out.printf("\tword ngrams = %d\n", jft.getWordNgrams());
    System.out.printf("\tloss name = %s\n", jft.getLossName());
    System.out.printf("\tmodel name = %s\n", jft.getModelName());
    System.out.printf("\tnumber of buckets = %d\n", jft.getBucket());
    System.out.printf("\tlabel prefix = %s\n\n", jft.getLabelPrefix());
  }

  /**
   * Test model unloading to release memory (Java's GC doesn't collect memory allocated by native
   * function calls).
   */
  @Test
  public void test09ModelUnloading() throws Exception {
    JFastText jft = new JFastText();
    System.out.println("Loading model ...");
    jft.loadModel("src/test/resources/models/supervised.model.bin");
    System.out.println("Unloading model ...");
    jft.unloadModel();
  }

  /** Test loading default language detection model. */
  @Test
  public void test10LoadDefaultLanguageDetectionModel() {
    JFastText jft = new JFastText();
    System.out.println("Loading model ...");
    jft.loadDefaultLanguageDetectionModel();

    System.out.println("Test English text...");
    List<String> predictions = jft.predict("English text", 3, 0.01f);
    for (String pred : predictions) {
      System.out.println(pred);
    }

    System.out.println("Test Chinese text...");
    List<ProbLabel> predictionProbas = jft.predictProba("中文文本", 3, 0.01f);
    for (ProbLabel pred : predictionProbas) {
      System.out.println(pred);
    }
    System.out.println("Unloading model ...");
    jft.unloadModel();
  }

  @Test
  public void test11GetSentenceVector() throws Exception {
    JFastText jft = new JFastText();
    jft.loadModel("src/test/resources/models/supervised.model.bin");
    String word =
        "Soccer is the word in American English. Football is the word used across the world.";
    List<Float> vec = jft.getSentenceVector(word);
    System.out.printf("Sentence embedding vector of '%s': %s\n", word, vec);
  }
}
