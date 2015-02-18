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
package mulan.dimensionalityReduction;

import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.data.MultiLabelInstances;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Instances;

/**
 * @author George Traianos
 * @author Grigorios Tsoumakas
 */
public class BinaryRelevanceAttributeEvaluator extends ASEvaluation implements AttributeEvaluator {

    /**
     * final scores for all attributes
     */
    private double[] scores;
    /**
     * The number of labels
     */
    int numLabels;
    /**
     * combination approach mode
     */
    private String CombApprMode;
    /**
     * normalization mode
     */
    private String NormMode;
    /**
     * attribute scoring based either on evaluation scores or ranking
     */
    private String ScoreMode;

    /**
     * a wrapper class for score-based attribute ranking
     */
    public class Rank implements Comparable {

        /**
         * score of the attribute
         */
        private double score;
        /**
         * index of the attribute
         */
        private int index;

        /**
         * constructor
         *
         * @param score the score to be given
         * @param index the index to be given
         */
        public Rank(double score, int index) {
            this.score = score;
            this.index = index;
        }

        /**
         * Returns the score of the attribute
         *
         * @return score of the attribute
         */
        public double getScore() {
            return score;
        }

        /**
         * Returns the index of the attribute
         *
         * @return index of the attribute
         */
        public int getIndex() {
            return index;
        }

        @Override
        public int compareTo(Object o) {
            if (score > ((Rank) o).score) {
                return 1;
            } else if (score < ((Rank) o).score) {
                return -1;
            } else {
                return 0;
            }
        }
    }

    /**
     * @param ase the evaluator type (weka type)
     * @param mlData multi-label instances for evaluation
     * @param combapp combination approach mode ("max", "avg", "min")
     * @param norm normalization mode ("dl", "dm", "none")
     * @param mode scoring mode ("eval", "rank")
     */
    public BinaryRelevanceAttributeEvaluator(ASEvaluation ase, MultiLabelInstances mlData, String combapp, String norm, String mode) {
        CombApprMode = combapp;
        NormMode = norm;
        ScoreMode = mode;

        numLabels = mlData.getNumLabels();
        try {
            int numAttributes = mlData.getFeatureIndices().length;
            double[][] evaluations = new double[numLabels][numAttributes];

            BinaryRelevanceTransformation brt = new BinaryRelevanceTransformation(mlData);
            for (int i = 0; i < numLabels; i++) {
                System.out.println("" + (i + 1) + "/" + (numLabels + 1));

                // create dataset
                Instances labelInstances = brt.transformInstances(i);

                // build evaluator
                ase.buildEvaluator(labelInstances);

                // evaluate features
                for (int j = 0; j < numAttributes; j++) {
                    evaluations[i][j] = ((AttributeEvaluator) ase).evaluateAttribute(j);
                }
            }

            // scoring of features
            scores = featureSelection(evaluations);
        } catch (Exception ex) {
            Logger.getLogger(BinaryRelevanceAttributeEvaluator.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * returns a ranking of attributes (where each attribute is represented by
     * its index)
     *
     * @param scores the attributes' scorelist
     * @return an ascending ranking of the attributes, based on their scores
     */
    public int[] rankAsc(double scores[]) {
        /*
         * create a table to hold each attribute's score and index
         */
        Rank r[] = new Rank[scores.length];

        for (int i = 0; i < r.length; i++) {
            r[i] = new Rank(scores[i], i);
        }

        /*
         * sort the table, thus resulting in ascending, score-based ranking
         */
        java.util.Arrays.sort(r);

        /*
         * create a ranking table containing only the attributes' indices
         */
        int ranking[] = new int[r.length];

        for (int i = 0; i < r.length; i++) {
            ranking[i] = r[i].getIndex();
        }

        return ranking;
    }

    /**
     * returns a ranking of attributes (where each attribute is represented by
     * its index)
     *
     * @param scores the attributes' scorelist
     * @return a descending ranking of the attributes, based on their scores
     */
    public int[] rankDesc(double scores[]) {
        int ranking[] = rankAsc(scores);
        int decr[] = new int[ranking.length];

        /*
         * receive the indices in reverse order, thus resulting in descending ranking
         */
        for (int i = 0; i < decr.length; i++) {
            decr[i] = ranking[(ranking.length - 1) - i];
        }

        return decr;
    }

    /**
     * orders the ranking scores according to their attributes' original indices
     *
     * @param ranking a rank table
     * @return the order of the ranking scores
     */
    public int[] order(int ranking[]) {
        int order[] = new int[ranking.length];

        for (int i = 0; i < ranking.length; i++) {
            order[ranking[i]] = i + 1;
        }

        return order;
    }

    /**
     * highest score combination approach
     *
     * @param scoreList all attributes' score lists
     * @param index the index of a specific attribute's score
     * @return the highest score achieved in any of the the input score lists
     */
    public double highest(double scoreList[][], int index) {
        double highest = scoreList[0][index];

        for (int i = 1; i < scoreList.length; i++) {
            highest = (scoreList[i][index] > highest ? scoreList[i][index] : highest);
        }

        return highest;
    }

    /**
     * lowest score combination approach
     *
     * @param scoreList all attributes' score lists
     * @param index the index of a specific attribute's score
     * @return the lowest score achieved in all of the input score lists
     */
    public double lowest(double scoreList[][], int index) {
        double lowest = scoreList[0][index];

        for (int i = 1; i < scoreList.length; i++) {
            lowest = (scoreList[i][index] < lowest ? scoreList[i][index] : lowest);
        }

        return lowest;
    }

    /**
     * average score combination approach
     *
     * @param scoreList all attributes' score lists
     * @param index the index of a specific attribute's score
     * @return the average score achieved in all the score lists
     */
    public double average(double scoreList[][], int index) {
        double sum = 0;

        for (int i = 0; i < scoreList.length; i++) {
            sum += scoreList[i][index];
        }

        return sum / scoreList.length;
    }

    /**
     * performs attribute selection
     *
     * @param evaluations evaluation scores
     * @return an array of scores for all attributes
     * @throws Exception when featureSelection fails
     */
    private double[] featureSelection(double evaluations[][]) throws Exception {
        // perform dm or dl
        if (!NormMode.equalsIgnoreCase("none")) {
            if (NormMode.equalsIgnoreCase("dm")) {
                for (int i = 0; i < evaluations.length; i++) {
                    evaluations[i] = dm(evaluations[i]);
                }
            } else if (NormMode.equalsIgnoreCase("dl")) {
                for (int i = 0; i < evaluations.length; i++) {
                    evaluations[i] = dl(evaluations[i]);
                }
            }
        }

        // to hold attributes' scores
        double tempScores[][] = new double[numLabels][];

        // rank based scoring of attributes
        if (ScoreMode.equalsIgnoreCase("rank")) {
            // perform ranking
            int ranks[][] = new int[numLabels][];

            for (int i = 0; i < evaluations.length; i++) {
                ranks[i] = rankDesc(evaluations[i]);
                order(ranks[i]);
            }

            // transform ranking into a score
            for (int i = 0; i < ranks.length; i++) {
                tempScores[i] = new double[ranks[i].length];

                for (int j = 0; j < ranks[i].length; j++) {
                    tempScores[i][j] = (ranks[i].length - 1) + ranks[i][j];
                }
            }
        } // evaluation score based scoring of attributes
        else if (ScoreMode.equalsIgnoreCase("eval")) {
            // simply copy the evaluation scores
            for (int i = 0; i < evaluations.length; i++) {
                tempScores[i] = java.util.Arrays.copyOf(evaluations[i], evaluations[i].length);
            }
        }

        // employ a combination approach method
        double combAppr[] = new double[tempScores[0].length];

        if (CombApprMode.equalsIgnoreCase("max")) // highest
        {
            for (int i = 0; i < combAppr.length; i++) {
                combAppr[i] = highest(tempScores, i);
            }
        } else if (CombApprMode.equalsIgnoreCase("min")) // lowest
        {
            for (int i = 0; i < combAppr.length; i++) {
                combAppr[i] = lowest(tempScores, i);
            }
        } else if (CombApprMode.equalsIgnoreCase("avg")) // average
        {
            for (int i = 0; i < combAppr.length; i++) {
                combAppr[i] = average(tempScores, i);
            }
        }

        // return the scores for all attributes
        return combAppr;
    }

    /**
     * calculates the norm of a vector
     *
     * @param vector a numeric array (as a vector)
     * @return the norm of the given vector
     */
    public static double norm(double vector[]) {
        double sumsq = 0;

        for (int i = 0; i < vector.length; i++) {
            sumsq += Math.pow(vector[i], 2);
        }

        return Math.sqrt(sumsq);
    }

    /**
     * normalizes an array (in the range of [0,1])
     *
     * @param array a numeric array
     */
    public static void normalize(double array[]) {
        /*
         * find the largest element
         */
        double max = array[0];

        for (int i = 1; i < array.length; i++) {
            max = (array[i] > max ? array[i] : max);
        }

        /*
         * normalize all elements
         */
        for (int j = 0; j < array.length; j++) {
            array[j] /= max;
        }
    }

    /**
     * divide by length (dl) normalization
     *
     * @param array a numeric array
     * @return a dl normalized copy of array
     */
    public static double[] dl(double array[]) {
        /*
         * a copy of the original array
         */
        double copy[] = java.util.Arrays.copyOf(array, array.length);

        /*
         * calculate the norm
         */
        double norm = norm(copy);

        /*
         * divide each element by the norm
         */
        for (int i = 0; i < copy.length; i++) {
            copy[i] /= norm;
        }

        return copy;
    }

    /**
     * divide by maximum (dm) normalization
     *
     * @param array a numeric array
     * @return a dm normalized copy of array
     */
    public static double[] dm(double array[]) {
        /*
         * a copy of the original array
         */
        double[] copy = java.util.Arrays.copyOf(array, array.length);

        /*
         * normalize the copy
         */
        normalize(copy);

        return copy;
    }

    /**
     * Evaluates an attribute
     *
     * @param attribute the attribute index
     * @return the evaluation
     * @throws Exception when evaluate Attribute fails
     */
    @Override
    public double evaluateAttribute(int attribute) throws Exception {
        return scores[attribute];
    }

    /**
     * Not supported
     *
     * @param data functionality is not supported yet
     * @throws Exception functionality is not supported yet
     */
    @Override
    public void buildEvaluator(Instances data) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }
}
