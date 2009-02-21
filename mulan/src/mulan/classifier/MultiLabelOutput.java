package mulan.classifier;

import java.util.Arrays;

/**
 * Class representing the output of a MultiLabelLearner.
 * This can be a bipartition of labels into true and false, a ranking of labels,
 * or an array of confidence values for each label.
 *
 * @author greg
 */
public class MultiLabelOutput {

    private boolean[] bipartition;
    private int[] ranking;
    private double[] confidences;

    public MultiLabelOutput() {}

    public MultiLabelOutput(boolean[] aBipartition) {
        bipartition = Arrays.copyOf(aBipartition, aBipartition.length);
    }

    public MultiLabelOutput(boolean[] aBipartition, double[] someConfidences) {
        this(aBipartition);
        confidences = Arrays.copyOf(someConfidences, someConfidences.length);
        ranking = ranksFromConfidences(someConfidences);
    }

    public boolean[] getBipartition() {
        return bipartition;
    }

    public boolean hasBipartition() {
        return (bipartition != null);
    }

    public int[] getRanking() {
        return ranking;
    }

    public boolean hasRanking() {
        return (ranking != null);
    }

    public double[] getConfidences() {
        return confidences;
    }

    public boolean hasConfidences() {
        return (confidences != null);
    }

    private int[] ranksFromConfidences(double[] confidences) {
        int[] reverseRanks = weka.core.Utils.stableSort(confidences);
        int[] ranks = new int[confidences.length];
        for (int i=0; i<confidences.length; i++) {
            ranks[i] = reverseRanks[confidences.length-1-i];
        }
        return ranks;
    }

}
