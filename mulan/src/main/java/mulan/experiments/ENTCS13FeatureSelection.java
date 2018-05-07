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
package mulan.experiments;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.data.MultiLabelInstances;
import mulan.dimensionalityReduction.BinaryRelevanceAttributeEvaluator;
import mulan.dimensionalityReduction.LabelPowersetAttributeEvaluator;
import mulan.dimensionalityReduction.Ranker;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * <p>Class replicating the experiment in <em>Newton Spolaôr, Everton Alvares
 * Cherman, Maria Carolina Monard, and Huei Diana Lee. 2013. A Comparison of
 * Multi-label Feature Selection Methods using the Problem Transformation
 * Approach. Electron. Notes Theor. Comput. Sci. 292 (March 2013), 135-151.</em>
 * </p>
 *
 * @author Newton Spolaôr (newtonspolaor@gmail.com)
 * @version 2013.07.25
 */
public class ENTCS13FeatureSelection {

    /**
     * Constant defining the ascending sort
     */
    private final static int ASCENDING = 0;
    /**
     * Constant defining the descending sort
     */
    private final static int DESCENDING = 1;

    /**
     * Main class. This command requires the specification of two filenames and
     * the feature selection method name as command-line arguments. These
     * arguments correspond respectively to the name of the XML and the ARFF
     * files specifying the actual dataset, according to the Mulan standard
     * (http://mulan.sourceforge.net/format.html), and the name of the feature
     * selection method to be used. The methods currently available are named
     * "RF-BR", "RF-LP", "IG-BR" and "IG-LP".
     *
     * @param args command line arguments
     */
    public static void main(String[] args) {

        try {
            /**
             * Feature ranking begins
             */
            String arffFilename = Utils.getOption("arff", args);
            String xmlFilename = Utils.getOption("xml", args);

            System.out.println("\nLoading the data set\n");
            MultiLabelInstances dataSet = new MultiLabelInstances(arffFilename, xmlFilename); //original dataset with all features

            System.out.println("\nBuilding the feature evaluator and the problem transformation approach: " + args[4] + "\n"); //args[4] has the name of the feature selection method to be applied

            ASEvaluation multiLabelFeatureSelectionMethod = buildMultiLabelFeatureSelection(args[4], dataSet);
            System.out.println("\nEvaluating features by " + args[4] + "\n");
            Ranker r = new Ranker();
            r.search((AttributeEvaluator) multiLabelFeatureSelectionMethod, dataSet);

            System.out.println("\nOutputting the sorted evaluated attribute list\n");
            double[][] sortedEvaluatedAttributeList = sortedEvaluatedAttributeList(dataSet.getFeatureIndices(), (AttributeEvaluator) multiLabelFeatureSelectionMethod);
            int numberFeatures = sortedEvaluatedAttributeList.length;
            for (int i = 0; i < numberFeatures; i++) {
                System.out.println(sortedEvaluatedAttributeList[i][1] + " " + (((int) (sortedEvaluatedAttributeList[i][0])) + 1)); //This line prints the feature index as Weka does: actual feature index + 1
            }

            System.out.println("\nSetting the number of features to be returned from ranking\n");
            int[] featureIndices = featureIndicesByThreshold(0.01, sortedEvaluatedAttributeList); /*Used in ENTCS2013 paper*/
            //int [] featureIndices = featureIndicesByKBest(10, sortedEvaluatedAttributeList); /*Alternative function. See function documentation*/
            //int [] featureIndices = featureIndicesbyTPercent(0.25, sortedEvaluatedAttributeList); /*Alternative function. See function documentation*/

            System.out.println("\nOutputting the indices of the chosen features\n");
            numberFeatures = featureIndices.length;
            for (int i = 0; i < numberFeatures; i++) {
                System.out.println(featureIndices[i] + 1); //This line prints the feature index as Weka does: actual feature index + 1
            }

            /**
             * Feature ranking ends
             */
            buildReducedMultiLabelDataset(featureIndices, dataSet); //optional

        } catch (Exception ex) {
            Logger.getLogger(ENTCS13FeatureSelection.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Initiates {@link weka.attributeSelection.ASEvaluation} given by a Weka
     * feature importance measure and a Mulan approach to deal with
     * {@link MultiLabelInstances}
     *
     * @param multiLabelFeatureSelectionMethod name of the multi-label feature
     * selection method ("RF-BR", "RF-LP", "IG-BR", "IG-LP)
     * @param dataSet original dataset with all features. This dataset should
     * not have any feature/label named "class"
     * @return an initialized {@link weka.attributeSelection.ASEvaluation} to
     * perform multi-label feature selection
     */
    public static ASEvaluation buildMultiLabelFeatureSelection(String multiLabelFeatureSelectionMethod, MultiLabelInstances dataSet) {
        if (multiLabelFeatureSelectionMethod.equalsIgnoreCase("RFBR") || multiLabelFeatureSelectionMethod.equalsIgnoreCase("RF-BR")) {
            return new BinaryRelevanceAttributeEvaluator(new ReliefFAttributeEval(), dataSet, "avg", "none", "eval");
        } else if (multiLabelFeatureSelectionMethod.equalsIgnoreCase("RFLP") || multiLabelFeatureSelectionMethod.equalsIgnoreCase("RF-LP")) {
            return new LabelPowersetAttributeEvaluator(new ReliefFAttributeEval(), dataSet);
        } else if (multiLabelFeatureSelectionMethod.equalsIgnoreCase("IGBR") || multiLabelFeatureSelectionMethod.equalsIgnoreCase("IG-BR")) {
            return new BinaryRelevanceAttributeEvaluator(new InfoGainAttributeEval(), dataSet, "avg", "none", "eval");
        } else if (multiLabelFeatureSelectionMethod.equalsIgnoreCase("IGLP") || multiLabelFeatureSelectionMethod.equalsIgnoreCase("IG-LP")) {
            return new LabelPowersetAttributeEvaluator(new InfoGainAttributeEval(), dataSet);
        }
        System.out.println("multiLabelFeatureSelectionMethod should be set on one of the allowed values");
        System.exit(1);
        return null;
    }

    /**
     * Given an array with feature indices, this method returns a double matrix
     * similar to the one returned by the method rankedAttributes from the Weka
     * {@link weka.attributeSelection.Ranker} class
     *
     * @param featureIndices an array of feature indices
     * @param attributeEval the attribute evaluator to guide the evaluation
     * @return a matrix of sorted attribute indices and evaluations
     */
    public static double[][] sortedEvaluatedAttributeList(int[] featureIndices, AttributeEvaluator attributeEval) {
        double[][] result = new double[featureIndices.length][2];

        try {
            for (int i = 0; i < featureIndices.length; i++) { //the feature indices after problem transformation range from 0 to featureIndices.length
                result[i][0] = featureIndices[i]; //actual feature index
                result[i][1] = attributeEval.evaluateAttribute(i);
            }
            sortAttributeRanking(result, 1, DESCENDING);
        } catch (Exception ex) {
            Logger.getLogger(ENTCS13FeatureSelection.class.getName()).log(Level.SEVERE, null, ex);
        }

        return result;
    }

    /**
     * Weka attribute ranking sorting
     *
     * @param ranking matrix to be sorted
     * @param index ranking used as index for sorting
     * @param sortOrder sorting can be ASCENDING or DESCENDING
     */
    public static void sortAttributeRanking(double[][] ranking, int index, int sortOrder) {

        if (sortOrder == DESCENDING) {
            for (int i = 0; i < ranking.length; i++) {
                int index_max = i;
                for (int j = i + 1; j < ranking.length; j++) {
                    if (ranking[j][index] > ranking[index_max][index]) {
                        index_max = j;
                    }
                }
                if (index_max != i) {
                    double auxRank = ranking[index_max][1];
                    double auxIndex = ranking[index_max][0];

                    ranking[index_max][0] = ranking[i][0];
                    ranking[index_max][1] = ranking[i][1];
                    ranking[i][1] = auxRank;
                    ranking[i][0] = auxIndex;
                }
            }
        } else if (sortOrder == ASCENDING) {
            for (int i = 0; i < ranking.length; i++) {
                int index_max = i;
                for (int j = i + 1; j < ranking.length; j++) {
                    if (ranking[j][index] < ranking[index_max][index]) {
                        index_max = j;
                    }
                }
                if (index_max != i) {
                    double auxRank = ranking[index_max][1];
                    double auxIndex = ranking[index_max][0];

                    ranking[index_max][0] = ranking[i][0];
                    ranking[index_max][1] = ranking[i][1];
                    ranking[i][1] = auxRank;
                    ranking[i][0] = auxIndex;
                }
            }
        }
    }

    /**
     * Selects all features from a sorted evaluated attribute list with
     * evaluation better than a threshold
     *
     * @param threshold the threshold limiting the results
     * @param sortedEvaluatedAttributeList feature ranking returned from
     * {@link Ranker}
     * @return the indices of the selected features
     */
    public static int[] featureIndicesByThreshold(double threshold, double[][] sortedEvaluatedAttributeList) {
        ArrayList<Integer> featureIndicesArray = new ArrayList<>(); //temporary object

        boolean end = false;
        int listIndex = 0;

        while (end == false) {
            if (sortedEvaluatedAttributeList[listIndex][1] > threshold) { //if the current feature has importance higher than the threshold
                featureIndicesArray.add((int) sortedEvaluatedAttributeList[listIndex][0]); //add the index of the current feature to the array
                listIndex++;
                if (listIndex == sortedEvaluatedAttributeList.length) //all features have importance higher than the threshold
                {
                    end = true;
                }
            } else { //the current feature and the remaining ones do not have importance higher than the threshold
                end = true;
            }
        }

        if (featureIndicesArray.isEmpty()) {
            System.out.println("No feature is better than the threshold");
            System.exit(1);
            return null;
        }

        int[] featureIndices = new int[featureIndicesArray.size()];
        for (int i = 0; i < featureIndices.length; i++) {
            featureIndices[i] = featureIndicesArray.get(i).intValue();
        }
        return featureIndices;
    }

    /**
     * Selects the k best features from a sorted evaluated attribute list
     *
     * @param k value in (0,M]
     * @param sortedEvaluatedAttributeList feature ranking returned from
     * {@link Ranker}
     * @return the indices of the selected features
     */
    public static int[] featureIndicesByKBest(int k, double[][] sortedEvaluatedAttributeList) {
        if ((k <= 0) || (k > sortedEvaluatedAttributeList.length)) {
            System.out.println("k should be a value in (0,M], where M is the number of features of the original dataset");
            System.exit(1);
        }

        int[] featureIndices = new int[k];

        for (int i = 0; i < k; i++) {
            featureIndices[i] = ((int) sortedEvaluatedAttributeList[i][0]); //add the index of the ith best feature to the array
        }

        return featureIndices;
    }

    /**
     * Selects the t% best features from a sorted evaluated attribute list
     *
     * @param t percentage of the number of features with value in (0,1)
     * @param sortedEvaluatedAttributeList feature ranking returned from
	 * {@link Ranker}
     * @return the indices of the selected features
     */
    public static int[] featureIndicesbyTPercent(double t, double[][] sortedEvaluatedAttributeList) {
        if ((t >= 1) || (t <= 0)) {
            System.out.println("t should be a value in (0,1)");
            System.exit(1);
        }

        int[] featureIndices = new int[(int) Math.round((double) t * sortedEvaluatedAttributeList.length)];

        for (int i = 0; i < featureIndices.length; i++) {
            featureIndices[i] = ((int) sortedEvaluatedAttributeList[i][0]); //add the index of the ith best feature to the array
        }

        return featureIndices;
    }

    /**
     * Builds {@link MultiLabelInstances} from the dataset reduced by feature
     * selection
     *
     * @param featureIndices the indices of the selected features
     * @param dataSet original dataset with all features
     */
    private static void buildReducedMultiLabelDataset(int[] featureIndices, MultiLabelInstances dataSet) {
        //System.out.println("\nBuilding the reduced dataset from the chosen features\n");
        int[] toKeep = new int[featureIndices.length + dataSet.getNumLabels()];
        System.arraycopy(featureIndices, 0, toKeep, 0, featureIndices.length);
        int[] labelIndices = dataSet.getLabelIndices();
        System.arraycopy(labelIndices, 0, toKeep, featureIndices.length, dataSet.getNumLabels());

        Remove filterRemove = new Remove();
        filterRemove.setAttributeIndicesArray(toKeep);
        filterRemove.setInvertSelection(true);
        try {
            filterRemove.setInputFormat(dataSet.getDataSet());
            Instances filtered = Filter.useFilter(dataSet.getDataSet(), filterRemove);
            MultiLabelInstances mlFiltered = new MultiLabelInstances(filtered, dataSet.getLabelsMetaData());
            // You can now work on the reduced multi-label dataset mlFiltered
        } catch (Exception ex) {
            Logger.getLogger(ENTCS13FeatureSelection.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}