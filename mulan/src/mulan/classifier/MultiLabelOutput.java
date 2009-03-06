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

    /**
     * Creates a new instance of {@link MultiLabelOutput}.
     * @param aBipartition bipartiton of labels
     * @throws IllegalArgumentException if aBipartitions is null.
     */
    public MultiLabelOutput(boolean[] aBipartition) {
    	if(aBipartition == null){
    		throw new IllegalArgumentException("The bipartitions is null.");
    	}
        bipartition = Arrays.copyOf(aBipartition, aBipartition.length);
    }

    /**
     * Creates a new instance of {@link MultiLabelOutput}.
     * @param aRanking ranking of labels
     * @throws IllegalArgumentException if aRanking is null
     */
    public MultiLabelOutput(int[] aRanking) {
        if (aRanking == null) {
    		throw new IllegalArgumentException("The ranking is null.");
    	}
        ranking = Arrays.copyOf(aRanking, aRanking.length);
    }

    /**
     * Creates a new instance of {@link MultiLabelOutput}.
     * @param aBipartition bipartition of labels
     * @param someConfidences values of labels
     * @throws IllegalArgumentException if either of input parameters is null
     * @throws IllegalArgumentException if dimension of bipartition and 
     * 									values does not match
     */
    public MultiLabelOutput(boolean[] aBipartition, double[] someConfidences) {
        this(aBipartition);
        if(someConfidences == null){
    		throw new IllegalArgumentException("The confidences is null.");
    	}
        if(aBipartition.length != someConfidences.length){
        	bipartition = null;
        	throw new IllegalArgumentException("The bipartitons and respective " +
        			"confidences dimansions does not match.");
        }
        confidences = Arrays.copyOf(someConfidences, someConfidences.length);
        ranking = ranksFromValues(someConfidences);
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

    public static int[] ranksFromValues(double[] values) {
        int[] temp = weka.core.Utils.stableSort(values);
        int[] ranks = new int[values.length];
        for (int i=0; i<values.length; i++)
            ranks[temp[i]] = values.length-i;
        return ranks;
    }

}
