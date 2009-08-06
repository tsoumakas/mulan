/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    MultiLabelOutput.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.classifier;

import java.util.Arrays;

/**
 * Class representing the output of a {@link MultiLabelLearner}.
 * This can be a bipartition of labels into <code>true</code> and <code>false</code>,
 * a ranking of labels, or an array of confidence values for each label.
 *
 * @author Grigorios Tsoumakas
 */
public class MultiLabelOutput {

    /** a bipartition of the labels into relevant and irrelevant */
    private boolean[] bipartition;

    /** the rank of each label, ranging from 1 to array length */
    private int[] ranking;

    /** the probability of each label being positive */
    private double[] confidences;

    /**
     * Creates a new instance of {@link MultiLabelOutput}.
     * @param bipartition bipartition of labels
     * @throws IllegalArgumentException if bipartitions is null.
     */
    public MultiLabelOutput(boolean[] bipartition) {
    	if(bipartition == null){
    		throw new IllegalArgumentException("The bipartitions is null.");
    	}
        this.bipartition = Arrays.copyOf(bipartition, bipartition.length);
    }

    /**
     * Creates a new instance of {@link MultiLabelOutput}.
     * @param ranking ranking of labels
     * @throws IllegalArgumentException if ranking is null
     */
    public MultiLabelOutput(int[] ranking) {
        if (ranking == null) {
    		throw new IllegalArgumentException("The ranking is null.");
    	}
        this.ranking = Arrays.copyOf(ranking, ranking.length);
    }

    /**
     * Creates a new instance of {@link MultiLabelOutput}. It creates a ranking
     * based on the probabilities and a bipartition based on a threshold for the probabilities.
     *
     * @param probabilities score of each label
     * @param threshold threshold to output bipartition based on probabilities
     * @throws IllegalArgumentException if probabilities is null
     */
    public MultiLabelOutput(double[] probabilities, double threshold) {
        if(probabilities == null){
    		throw new IllegalArgumentException("The probabilities array is null.");
    	}
        confidences = probabilities;
        ranking = ranksFromValues(probabilities);
        bipartition = new boolean[probabilities.length];
        for (int i=0; i<probabilities.length; i++)
            if (probabilities[i] >= threshold)
                bipartition[i] = true;
    }

    /**
     * Creates a new instance of {@link MultiLabelOutput}.
     * @param bipartition bipartition of labels
     * @param someConfidences values of labels
     * @throws IllegalArgumentException if either of the input parameters is null or
     * their dimensions do not match
     */
    public MultiLabelOutput(boolean[] bipartition, double[] someConfidences) {
        this(bipartition);
        if(someConfidences == null){
    		throw new IllegalArgumentException("The confidences is null.");
    	}
        if(bipartition.length != someConfidences.length){
        	this.bipartition = null;
        	throw new IllegalArgumentException("The dimensions of the bipartition " +
                    " and confidences arrays do not match.");
        }
        confidences = Arrays.copyOf(someConfidences, someConfidences.length);
        ranking = ranksFromValues(someConfidences);
    }

    /**
     * Gets bipartition of labels. 
     * @return
     */
    public boolean[] getBipartition() {
        return bipartition;
    }

    /**
     * Determines whether the {@link MultiLabelOutput} has bipartition of labels.
     * @return <code>true</code> if has bipartition; otherwise <code>false</code>
     */
    public boolean hasBipartition() {
        return (bipartition != null);
    }

    /**
     * Gets ranking of labels.
     * @return
     */
    public int[] getRanking() {
        return ranking;
    }

    /**
     * Determines whether the {@link MultiLabelOutput} has ranking of labels.
     * @return <code>true</code> if has ranking; otherwise <code>false</code>
     */
    public boolean hasRanking() {
        return (ranking != null);
    }

    /**
     * Gets confidences of labels.
     * @return
     */
    public double[] getConfidences() {
        return confidences;
    }

    /**
     * Determines whether the {@link MultiLabelOutput} has confidences of labels.
     * @return <code>true</code> if has confidences; otherwise <code>false</code>
     */
    public boolean hasConfidences() {
        return (confidences != null);
    }

    /**
     * Creates a ranking form specified values/confidences.
     * 
     * @param values the values/confidences to be converted to ranking
     * @return the ranking of given values/confidences
     */
    public static int[] ranksFromValues(double[] values) {
        int[] temp = weka.core.Utils.stableSort(values);
        int[] ranks = new int[values.length];
        for (int i=0; i<values.length; i++)
            ranks[temp[i]] = values.length-i;
        return ranks;
    }

}
