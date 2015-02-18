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
package mulan.classifier.meta;

import java.util.*;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.ConditionalDependenceIdentifier;
import mulan.data.LabelPairsDependenceIdentifier;
import mulan.data.LabelsPair;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.TechnicalInformation;

/**
 <!-- globalinfo-start -->
 * A class for gathering several different SubsetLearners into a composite ensemble model. &lt;br&gt; &lt;br&gt; The label set partitions for participation in ensemble are selected using their dependence weight from the large number of randomly generated possible partitions. The type of the learned dependencies is determined by the {&#64;link mulan.data.LabelPairsDependenceIdentifier} supplied to the class constructor. Two strategies for selecting ensemble partitions exists: (1) to select the highly weighted ones and (2) to select most different from the highly weighted ones. The strategy to be used is determined by the {&#64;link #selectDiverseModels} parameter which is 'true' by default.<br>
 * <br>
 * For more information, see<br>
 * <br>
 * Lena Tenenboim-Chekina, Lior Rokach,, Bracha Shapira: Identification of Label Dependencies for Multi-label Classification. In: , Haifa, Israel, 53--60, 2010.
 * <br>
 <!-- globalinfo-end --> 
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{LenaTenenboim-Chekina2010,
 *    address = {Haifa, Israel},
 *    author = {Lena Tenenboim-Chekina, Lior Rokach, and Bracha Shapira},
 *    pages = {53--60},
 *    title = {Identification of Label Dependencies for Multi-label Classification},
 *    volume = {Proc. ICML 2010 Workshop on Learning from Multi-Label Data (MLD'10},
 *    year = {2010}
 * }
 * </pre>
 * <br>
 <!-- technical-bibtex-end -->
 * 
 * @author Lena Chekina (lenat@bgu.ac.il)
 * @version 30.11.2010
 */
public class EnsembleOfSubsetLearners extends MultiLabelMetaLearner {

    /**
     * An array of ensemble models
     */
    SubsetLearner[] ensembleModels;
    /**
     * The number of models in the ensemble
     */
    int numModels = 10;
    /**
     * The threshold for ensemble voting
     */
    double threshold = 0.5;
    /**
     * Defines the type of dependence identification process.
     */
    LabelPairsDependenceIdentifier dependenceIdentifier;
    /**
     * Base Classifier that will be used for single label training and
     * predictions
     */
    Classifier singleLabelLearner;
    /**
     * Select most different from the highly weighted partitions
     */
    boolean selectDiverseModels = true;
    /**
     * Disable SubsetLearner caching mechanism
     */
    boolean useSubsetcache = false;
    /**
     * Seed for replication of random experiments
     */
    private int seed = 1;
    /**
     * Random number generator
     */
    private Random rnd;
    /**
     * Number of randomly generated possible label set partitions
     */
    private static int numOfRandomPartitions = 50000;
    /**
     * Number of highly weighted partitions used for selecting the 'enough'
     * different among them. Used when {@link #selectDiverseModels} is true.
     */
    private static int numOfPartitionsForDiversity = 100;
    /**
     * Parameter used to dynamically define the threshold of 'enough' different
     * partition. Used when
     * {@link #selectDiverseModels} is true.
     */
    private static double dynamicDiversityThreshold = 0.2;

    /**
     * Default constructor. Can be used for accessing class utility methods.
     *
     */
    public EnsembleOfSubsetLearners() {
        this(new BinaryRelevance(new J48()), new J48(), new ConditionalDependenceIdentifier(new J48()), 10);
    }

    /**
     * Initialize EnsembleOfSubset with multilabel and single label learners, a
     * method for labels dependence identification and number of models to
     * constitute the ensemble.
     *
     * @param aMultiLabelLearner the learner for multilabel classification
     * @param aSingleLabelLearner the learner for single label classification
     * @param aDependenceIdentifier the method for label pairs dependence
     * identification
     * @param models the number of models
     */
    public EnsembleOfSubsetLearners(MultiLabelLearner aMultiLabelLearner,
            Classifier aSingleLabelLearner, LabelPairsDependenceIdentifier aDependenceIdentifier,
            int models) {
        super(aMultiLabelLearner);
        singleLabelLearner = aSingleLabelLearner;
        dependenceIdentifier = aDependenceIdentifier;
        numModels = models;
        rnd = new Random(seed);
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.INPROCEEDINGS);
        result.setValue(TechnicalInformation.Field.AUTHOR,
                "Lena Tenenboim-Chekina, Lior Rokach, and Bracha Shapira");
        result.setValue(TechnicalInformation.Field.TITLE,
                "Identification of Label Dependencies for Multi-label Classification");
        result.setValue(TechnicalInformation.Field.VOLUME,
                "Proc. ICML 2010 Workshop on Learning from Multi-Label Data (MLD'10");
        result.setValue(TechnicalInformation.Field.YEAR, "2010");
        result.setValue(TechnicalInformation.Field.PAGES, "53--60");
        result.setValue(TechnicalInformation.Field.ADDRESS, "Haifa, Israel");
        return result;
    }

    /**
     * Builds an ensemble of Subset models. Label set partitions for ensemble
     * are selected based on the set {@link #dependenceIdentifier} and value of
     * the {@link #selectDiverseModels} parameter.
     *
     * @param trainingData the training data set
     * @throws Exception if learner model was not created successfully
     */
    @Override
    protected void buildInternal(MultiLabelInstances trainingData) throws Exception {
        int totalSubsets = 0;
        List<LabelSubsetsWeight> pairsList = createLabelSetPartitions(trainingData);
        ensembleModels = new SubsetLearner[numModels];
        for (int m = 0; m < numModels; m++) {
            LabelSubsetsWeight pair = pairsList.get(m);
            int[][] comb = pair.getSubsets();
            ensembleModels[m] = new SubsetLearner(comb, singleLabelLearner);
            ensembleModels[m].setUseCache(useSubsetcache);
            totalSubsets = totalSubsets + comb.length;
            debug("Building model" + m + ":" + partitionToString(comb) + " weight="
                    + pair.getValue());
            ensembleModels[m].build(trainingData);
        }
        debug("Total Subsets  =" + totalSubsets + '\n');
    }

    /**
     * Makes classification prediction using constructed ensemble of Subset
     * models. For each label models' votes are summarized, and confidence is
     * computed as average of positive predictions.
     *
     * @param instance the data instance to predict on
     * @return the {@link mulan.classifier.MultiLabelOutput} classification
     * prediction for the instance.
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        int[] sumVotes = new int[numLabels];
        for (int i = 0; i < numModels; i++) { // gather models' votes
            MultiLabelOutput ensembleMLO = ensembleModels[i].makePrediction(instance);
            boolean[] bip = ensembleMLO.getBipartition();
            for (int j = 0; j < sumVotes.length; j++) { // summarize votes for each label
                sumVotes[j] += bip[j] ? 1 : 0;
            }
        }
        double[] confidence = new double[numLabels];
        for (int j = 0; j < sumVotes.length; j++) { // convert votes to confidence
            confidence[j] = (double) sumVotes[j] / (double) numModels;
        }
        return new MultiLabelOutput(confidence, threshold);
    }

    /**
     * Creates the specified number of randomly generated possible label set
     * partitions consisting of the specified number of labels..
     *
     * @param numLabels - number of labels
     * @param numSets - number of random partitions to generate
     * @return a list of generated partitions
     */
    public List<int[][]> createRandomSets(int numLabels, int numSets) {
        List<int[][]> sets;
        int n = 2 * numLabels - 1; // values from 0 to numLabels-1 will indicate a label and
        // values from numLabels to n-1 will indicate a separator of two groups
        int[][] permutations = GenerateRandomPermutations(n, numSets);
        sets = convertToSets(permutations, numLabels);
        return sets;
    }

    /**
     * Returns a string representation of the labels partition.
     *
     * @param partition - a label set partition
     * @return a string representation of the labels partition
     */
    public static String partitionToString(int[][] partition) {
        StringBuilder result = new StringBuilder();
        for (int[] aGroup : partition) {
            result.append(Arrays.toString(aGroup));
            result.append(", ");
        }
        return result.toString();
    }

    /*
     * Creates and selects partitions to constitute the ensemble.
     *
     * @param trainingData the training data set
     *
     * @return a list of selected label set partitions
     */
    private List<LabelSubsetsWeight> createLabelSetPartitions(MultiLabelInstances trainingData) {
        List<LabelSubsetsWeight> selectedSets; // a list of selected partitions
        LabelsPair[] labelPairs = dependenceIdentifier.calculateDependence(trainingData); // get a list of labels pairs along
        //  with their dependence scores
        double criticalValue = dependenceIdentifier.getCriticalValue();
        double[][] weightsMatrix = createDependenceWeightsMatrix(labelPairs, criticalValue,
                numLabels, true); // compute normalized dependence scores of all label pairs
        List<int[][]> randomPartitions = createRandomSets(numLabels, numOfRandomPartitions); // create random label set partitions
        ArrayList<LabelSubsetsWeight> weightedSets = setWeights(randomPartitions, weightsMatrix); // compute 'dependence' score
        //  of each random partition
        Collections.sort(weightedSets, Collections.reverseOrder()); // sort partitions in descending
        // order of the 'dependence' score
        List<LabelSubsetsWeight> distinctSets = getDistinctSets(weightedSets); // remove same partitions
        // of labels (as were randomly generated)
        if (selectDiverseModels) { // for the diverse version of the algorithm
            int numForDiversity = Math.min(distinctSets.size(), numOfPartitionsForDiversity); // number of high scored partition to consider
            List<LabelSubsetsWeight> highestSets = getHighOrdered(distinctSets, numForDiversity); // get 'numForDiversity' high scored partitions
            selectedSets = selectByDiversity(highestSets);
        } else { // take 'numModels' high scored partitions
            selectedSets = getHighOrdered(distinctSets, numModels);
        }
        return selectedSets;
    }

    /**
     * Creates a matrix containing dependence score for each labels pair.
     *
     * @param pairsList the list of labels pairs and their dependence scores
     * @param critical the statistic critical value
     * @param n the size of the matrix (i.e. the number of labels)
     * @param normalized indicates if to apply normalization using chiCritical
     * value
     * @return a matrix containing dependence score for each labels pair.
     */
    private static double[][] createDependenceWeightsMatrix(LabelsPair[] pairsList,
            double critical, int n, boolean normalized) {
        double[][] matrix = new double[n][n];
        for (LabelsPair pair : pairsList) {
            Double value = pair.getScore();
            int[] data = pair.getPair();
            if (normalized) {
                if (data[0] < data[1]) { // fill matrix above the diagonal
                    matrix[data[0]][data[1]] = value - critical;
                } else {
                    matrix[data[1]][data[0]] = value - critical;
                }
            } else {
                if (data[0] < data[1]) {
                    matrix[data[0]][data[1]] = value;
                } else {
                    matrix[data[1]][data[0]] = value;
                }
            }
        }
        return matrix;
    }

    /**
     * Generates random permutations of values from 0 to n-1.
     *
     * @param n indicates the length of a permutation: each permutation will
     * contain 'n' numbers from 0 to n-1 (inclusive)
     * @param num the number of permutations to be generated
     * @return arrays of random permutations of values from 0 to n-1.
     */
    private int[][] GenerateRandomPermutations(int n, int num) {
        int[][] permutations = new int[num][];
        int[] a = initialize(n); // initialization
        for (int i = 0; i < num; i++) {
            int[] rand = randomize(a);
            permutations[i] = rand;
        }
        return permutations;
    }

    /**
     * Fills an array with values from 0 to n-1
     *
     * @param n the array length
     * @return an array with values from 0 to n-1
     */
    private static int[] initialize(int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        return a;
    }

    /**
     * Randomly mixes the values of an array .
     *
     * @param a the array which values to be mixed
     * @return an array with randomly mixed values
     */
    private int[] randomize(int[] a) {
        int[] b = a.clone();
        for (int k = b.length - 1; k > 0; k--) {
            int w = (int) Math.floor(rnd.nextDouble() * (k + 1));
            int temp = b[w];
            b[w] = b[k];
            b[k] = temp;
        }
        return b;
    }

    /**
     * Converts random permutations to list of various label set partitions.
     *
     * @param permutations the arrays of permutations
     * @param numLabels the number of labels
     * @return a list of various label set partitions
     */
    private static List<int[][]> convertToSets(int[][] permutations, int numLabels) {
        List<int[][]> sets = new ArrayList<int[][]>();
        for (int[] permutation : permutations) {
            List<Integer[]> groupsList = extractGroups(permutation, numLabels);
            sets.add(createSet(groupsList));
        }
        return sets;
    }

    /**
     * Converts a random permutation to list of label subsets.
     *
     * @param permutation the array of permutation
     * @param numLabels the number of labels
     * @return a list of label subsets
     */
    private static List<Integer[]> extractGroups(int[] permutation, int numLabels) {
        List<Integer[]> groupsList = new ArrayList<Integer[]>();
        List<Integer> group = new ArrayList<Integer>();
        for (int value : permutation) {
            if (value < numLabels) { // if value indicates label index
                group.add(value);
            } else { // the value indicates a separator
                if (group.size() > 0) {
                    Integer gr[] = group.toArray(new Integer[group.size()]);
                    groupsList.add(gr);
                    group = new ArrayList<Integer>();
                }
            }
        }
        if (group.size() > 0) { // add the latest group
            Integer gr[] = group.toArray(new Integer[group.size()]);
            groupsList.add(gr);
        }
        return groupsList;
    }

    /**
     * Converts a list of arrays (i.e label subsets) into two-dimensional array
     * representing a label set partitioning.
     *
     * @param groupsList the list of label subsets
     * @return arrays of label set partitions
     */
    private static int[][] createSet(List<Integer[]> groupsList) {
        int numGroups = groupsList.size();
        int[][] sets = new int[numGroups][];
        Integer[][] sets2 = groupsList.toArray(new Integer[groupsList.size()][]);
        for (int i = 0; i < sets2.length; i++) { // convert to primitives
            sets[i] = new int[sets2[i].length];
            for (int j = 0; j < sets2[i].length; j++) {
                sets[i][j] = sets2[i][j];
            }
        }
        return sets;
    }

    /**
     * Computes 'dependence' score of each partition and store it in a list of
     * weighted subsets.
     *
     * @param partitions the label set partitions
     * @param weightsMatrix the matrix containing dependence score for each
     * labels pair
     * @return a list of weighted label subsets
     */
    private ArrayList<LabelSubsetsWeight> setWeights(List<int[][]> partitions,
            double[][] weightsMatrix) {
        ArrayList<LabelSubsetsWeight> weightedList = new ArrayList<LabelSubsetsWeight>(partitions.size());
        for (int[][] partition : partitions) {
            Double weight = computeWeight(partition, weightsMatrix, numLabels); // compute 'dependence' score of a partition
            LabelSubsetsWeight p = new LabelSubsetsWeight(weight, partition); // storing the weight for a partition in an intended class
            weightedList.add(p);
        }
        return weightedList;
    }

    /**
     * Computes 'dependence' score of a partition.
     *
     * @param partition the label set partition
     * @param weightsMatrix the matrix containing dependence score for each
     * labels pair
     * @param numLabels the number of labels
     * @return value indicating a 'dependence' score of the partition
     */
    private static Double computeWeight(int[][] partition, double[][] weightsMatrix, int numLabels) {
        double[][] matrix = deepClone(weightsMatrix);
        Double weight = 0.0;
        TreeSet<Integer> ind = new TreeSet<Integer>(); // a set of labels in other groups
        TreeSet<Integer> dep = new TreeSet<Integer>(); // a set of labels in the same group
        for (int[] aGroup : partition) {
            for (int aLabel : aGroup) { // for each label
                for (int bLabel : aGroup) { // all labels from the same group add to dep set
                    dep.add(bLabel);
                }
                for (int n = 0; n < numLabels; n++) { // all other labels add to ind
                    if (!dep.contains(n)) {
                        ind.add(n);
                    }
                }
                weight = weight + weightOf(dep, aLabel, matrix); // summing up the scores of all pairs
                // whose labels are in the same group
                weight = weight - weightOf(ind, aLabel, matrix); // subtracting the scores of all pairs
                // whose labels are in different groups
            }
        }
        return weight;
    }

    /**
     * Creates a deep copy of two-dimensional matrix
     *
     * @param matrix the matrix to copy
     * @return a deep opy of the matrix
     */
    private static double[][] deepClone(double[][] matrix) {
        double[][] m = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            m[i] = matrix[i].clone();
        }
        return m;
    }

    /**
     * Summarizes the weights of pairs of the label with other labels.
     *
     * @param otherLabels the set of labels which pair-weight with the specified
     * label should be summarized
     * @param label the label which pair-weight with other labels should be
     * summarized
     * @param matrix the matrix containing dependence score for each labels pair
     * @return sum of the weights of pairs of the label with each one of the
     * other labels
     */
    private static Double weightOf(TreeSet<Integer> otherLabels, int label, double[][] matrix) {
        Double w = 0.0;
        for (Integer l2 : otherLabels) {
            if (label < l2) { // getting values from above the diagonal of the matrix
                w = w + matrix[label][l2];
                matrix[label][l2] = 0; // each entry should be counted only once
            } else {
                w = w + matrix[l2][label];
                matrix[l2][label] = 0;
            }
        }
        return w;
    }

    /**
     * Filtering out identical label set partitions. An assumption is that
     * partitions with the same weight are identical.
     *
     * @param orderedList the list of weighted label subsets ordered by subset
     * weight
     * @return a list of distinct weighted label subsets
     */
    private static List<LabelSubsetsWeight> getDistinctSets(List<LabelSubsetsWeight> orderedList) {
        List<LabelSubsetsWeight> distinct = new ArrayList<LabelSubsetsWeight>();
        long v = 0;
        for (LabelSubsetsWeight subset : orderedList) {
            long value = subset.getValue().longValue();
            if (v != value) {
                distinct.add(subset);
            }
            v = value;
        }
        return distinct;
    }

    /**
     * Returns the specified number of highly weighted partitions.
     *
     * @param orderedList the list of weighted label subsets ordered by subset
     * weight in descending order
     * @param number the number of partitions to return
     * @return the specified number of highly weighted partitions.
     */
    private static List<LabelSubsetsWeight> getHighOrdered(List<LabelSubsetsWeight> orderedList,
            int number) {
        List<LabelSubsetsWeight> highest = new ArrayList<LabelSubsetsWeight>();
        int count = 0;
        for (LabelSubsetsWeight subset : orderedList) {
            highest.add(subset);
            count++;
            if (count == number) {
                return highest;
            }
        }
        return highest;
    }

    /**
     * Computes a distance between the two label set partitions.
     *
     * @param set1 the label set partition 1
     * @param set2 the label set partition 2
     * @param numLabels the number of labels
     * @return a distance between the two label set partitions
     */
    private static int distance(LabelSubsetsWeight set1, LabelSubsetsWeight set2, int numLabels) {
        int dist = 0;
        int[][] set1Matrix = matrixRepresentation(set1.getSubsets(), numLabels);
        int[][] set2Matrix = matrixRepresentation(set2.getSubsets(), numLabels);
        for (int i = 0; i < numLabels; i++) {
            for (int j = i + 1; j < numLabels; j++) {
                if ((set1Matrix[i][j] == 0 && set2Matrix[i][j] == 1)
                        || (set1Matrix[i][j] == 1 && set2Matrix[i][j] == 0)) {
                    dist++;
                }
            }
        }
        return dist;
    }

    /**
     * Represents a partition as a matrix: if two labels are in the same group
     * the relatesd entry of the matrix is 1, otherwise 0.
     *
     * @param set the label set partition to be represented by a matrix
     * @param numLabels the number of labels
     * @return a matrix representing the partition
     */
    private static int[][] matrixRepresentation(int[][] set, int numLabels) {
        int[][] setMatrix = new int[numLabels][numLabels];
        for (int[] anArray : setMatrix) { // initialize with 0
            Arrays.fill(anArray, 0);
        }
        for (int[] aGroup : set) {
            for (int j = 0; j < aGroup.length; j++) {
                int l1 = aGroup[j]; // for each element
                for (int k = j + 1; k < aGroup.length; k++) {
                    // set 1 for all elements in the same group
                    int l2 = aGroup[k];
                    setMatrix[l1][l2] = 1;
                    setMatrix[l2][l1] = 1;
                }
            }
        }
        return setMatrix;
    }

    /**
     * Selects most different from the highly weighted partitions. This method
     * is used when the
     * {@link #selectDiverseModels} parameter is 'true'.
     *
     * @param sets the list of weighted label subsets ordered by subset weight
     * in descending order
     * @return most different from the highly weighted label set partitions
     */
    private List<LabelSubsetsWeight> selectByDiversity(List<LabelSubsetsWeight> sets) {
        List<LabelSubsetsWeight> selected = new ArrayList<LabelSubsetsWeight>();
        selected.add(sets.get(0)); // add the first - the highly weighted one
        for (int i = 1; i < numModels; i++) { // for selecting each next partition:
            // 1 - compute min distance from the rest (or all) in sets to that in selected
            SubsetsDistance[] minDistToSelected = minDistToSelected(sets, selected);
            // 2 - select those with highest minimal distance (and among them highest set ids)
            Arrays.sort(minDistToSelected);
            // the secondary sort by set id is preserved here
            int endIndx = minDistToSelected.length - 1;
            int startIndx = endIndx - (int) (minDistToSelected.length * dynamicDiversityThreshold);
            SubsetsDistance[] differentSets = Arrays.copyOfRange(minDistToSelected, startIndx,
                    endIndx);
            Arrays.sort(differentSets, new IdComparator()); // 3 - take one with the highest score
            // (highest score => minimal id)
            Integer setId = differentSets[0].getSubsetsId();
            LabelSubsetsWeight selectedSet = sets.get(setId);
            selected.add(selectedSet);
        }
        return selected;
    }

    /**
     * Computes the minimal distance from each labels set partition to the
     * "selected" partitions. If a partition is within the "selected" it's
     * minimal distance is 0.
     *
     * @param allSets the list of all label set partitions
     * @param selected the list of selected label set partitions
     * @return an array containing the minimal distance from each label set
     * partition to the "selected" partitions
     */
    private SubsetsDistance[] minDistToSelected(List<LabelSubsetsWeight> allSets,
            List<LabelSubsetsWeight> selected) {
        // an array of minimal distances from each partition to those in selected
        SubsetsDistance[] minDists = new SubsetsDistance[allSets.size()];
        // an array of distances from a partition to each one in selected
        int dists[] = new int[selected.size()];
        int i = 0;
        for (LabelSubsetsWeight set1 : allSets) {
            int j = 0;
            for (LabelSubsetsWeight set2 : selected) {
                int d = distance(set1, set2, numLabels);
                dists[j++] = d;
            }
            Arrays.sort(dists);
            int min = dists[0]; // get the minimal distance
            SubsetsDistance p = new SubsetsDistance(i, min);
            minDists[i++] = p; // add partition id and its minimal distance to the resultant array
        }
        return minDists;
    }

    /**
     * A class for handling partitions and their dependence weights. The natural
     * order of instances of this class is according to the natural order of the
     * partition dependence weight.
     */
    private class LabelSubsetsWeight implements Comparable, Cloneable {

        int[][] subsets;
        Double value;

        /**
         * Initialize a LabelSubsetsWeight object using labels set partition and
         * its weight.
         *
         * @param v the partition dependence weight
         * @param comb the label set partition
         */
        public LabelSubsetsWeight(double v, int[][] comb) {
            subsets = comb;
            value = v;
        }

        /**
         * @return the label set partition
         */
        public int[][] getSubsets() {
            return subsets;
        }

        /**
         * @return the partition dependence weight
         */
        public Double getValue() {
            return value;
        }

        /**
         * Set the partition dependence weight to the specified value.
         *
         * @param value the partition dependence weight
         */
        public void setValue(Double value) {
            this.value = value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            LabelSubsetsWeight pair = (LabelSubsetsWeight) o;
            return !(value != null ? !value.equals(pair.value) : pair.value != null);
        }

        public int compareTo(Object otherPair) {
            if (otherPair == null) {
                throw new NullPointerException();
            }
            if (!(otherPair instanceof LabelSubsetsWeight)) {
                throw new ClassCastException("Invalid object");
            }
            Double value = ((LabelSubsetsWeight) otherPair).getValue();
            if (this.getValue() > value) {
                return 1;
            } else if (this.getValue() < value) {
                return -1;
            } else {
                return 0;
            }
        }

        @Override
        public int hashCode() {
            return value != null ? value.hashCode() : 0;
        }
    }

    /**
     * A class for handling partitions and their distance values using partition
     * order identifiers. The natural order of instances of this class is
     * according to the natural order of partition distance values.
     */
    private class SubsetsDistance implements Comparable {

        Integer subsetsId;
        Integer value;

        /**
         * Initialize a SubsetsDistance object using the partition identifier
         * and its distance value.
         *
         * @param _id the partition identifier
         * @param v the partition's distance
         */
        public SubsetsDistance(int _id, int v) {
            subsetsId = _id;
            value = v;
        }

        /**
         * @return the partition's identifier
         */
        public Integer getSubsetsId() {
            return subsetsId;
        }

        /**
         * @return the partition's distance
         */
        public int getValue() {
            return value;
        }

        /**
         * Set the partition distance to the specified value
         *
         * @param value the partition's distance value
         */
        public void setValue(int value) {
            this.value = value;
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            SubsetsDistance pair = (SubsetsDistance) o;
            return !(value != null ? !value.equals(pair.value) : pair.value != null);
        }

        public int compareTo(Object otherPair) {
            if (otherPair == null) {
                throw new NullPointerException();
            }
            if (!(otherPair instanceof SubsetsDistance)) {
                throw new ClassCastException("Invalid object");
            }
            Integer value = ((SubsetsDistance) otherPair).getValue();
            if (this.getValue() > value) {
                return 1;
            } else if (this.getValue() < value) {
                return -1;
            } else {
                return 0;
            }
        }

        @Override
        public int hashCode() {
            return value != null ? value.hashCode() : 0;
        }
    }

    /**
     * A comparator class for comparing SubsetsDistance objects by partition
     * identifier values.
     */
    private class IdComparator implements Comparator {

        public int compare(Object o1, Object o2) {
            if (o1 instanceof SubsetsDistance && o2 instanceof SubsetsDistance) {
                SubsetsDistance s1 = (SubsetsDistance) o1;
                SubsetsDistance s2 = (SubsetsDistance) o2;
                Integer i1 = s1.getSubsetsId();
                Integer i2 = s2.getSubsetsId();
                return i1.compareTo(i2);
            }
            return 0;
        }
    }

    /**
     * 
     * @param rnd random generator
     */
    public void setRnd(Random rnd) {
        this.rnd = rnd;
    }

    /**
     * 
     * @param threshold the threshold for ensemble voting
     */
    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    /**
     * 
     * @param dependenceIdentifier the type of dependence identification process
     */
    public void setDependenceIdentifier(LabelPairsDependenceIdentifier dependenceIdentifier) {
        this.dependenceIdentifier = dependenceIdentifier;
    }

    /**
     * 
     * @param x seed value
     */
    public void setSeed(int x) {
        seed = x;
        rnd = new Random(seed);
    }

    /**
     * 
     * @param models the number of models
     */
    public void setNumModels(int models) {
        numModels = models;
    }

    /**
     * 
     * @return Number of models
     */
    public int getNumModels() {
        return numModels;
    }

    /**
     * 
     * @return Most different from the highly weighted partitions
     */
    public boolean isSelectDiverseModels() {
        return selectDiverseModels;
    }

    /**
     * 
     * @param selectDiverseModels whether to select most different from the highly weighted partitions
     */
    public void setSelectDiverseModels(boolean selectDiverseModels) {
        this.selectDiverseModels = selectDiverseModels;
    }

    /**
     * 
     * @param useSubsetcache whether to use Subset caching
     */
    public void setUseSubsetLearnerCache(boolean useSubsetcache) {
        this.useSubsetcache = useSubsetcache;
    }

    /**
     * 
     * @param numOfRandomPartitions number of randomly generated possible label set partitions
     */
    public static void setNumOfRandomPartitions(int numOfRandomPartitions) {
        EnsembleOfSubsetLearners.numOfRandomPartitions = numOfRandomPartitions;
    }

    /**
     * 
     * @param numOfPartitionsForDiversity  number of highly weighted partitions used for selecting the 'enough' different among them
     */
    public static void setNumOfPartitionsForDiversity(int numOfPartitionsForDiversity) {
        EnsembleOfSubsetLearners.numOfPartitionsForDiversity = numOfPartitionsForDiversity;
    }

    /**
     * 
     * @param dynamicDiversityThreshold parameter used to dynamically define the threshold of 'enough' different
     * partition
     */
    public static void setDynamicDiversityThreshold(double dynamicDiversityThreshold) {
        EnsembleOfSubsetLearners.dynamicDiversityThreshold = dynamicDiversityThreshold;
    }
    
    public String globalInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("A class for gathering several different SubsetLearners ");
        sb.append("into a composite ensemble model. <br> <br> The label set ");
        sb.append("partitions for participation in ensemble are selected ");
        sb.append("using their dependence weight from the large number of ");
        sb.append("randomly generated possible partitions. The type of the ");
        sb.append("learned dependencies is determined by the ");
        sb.append("{@link mulan.data.LabelPairsDependenceIdentifier} supplied");
        sb.append(" to the class constructor. Two strategies for selecting ");
        sb.append("ensemble partitions exists: (1) to select the highly ");
        sb.append("weighted ones and (2) to select most different from the ");
        sb.append("highly weighted ones. The strategy to be used is ");
        sb.append("determined by the {@link #selectDiverseModels} parameter ");
        sb.append("which is 'true' by default.\n\nFor more information, ");
        sb.append("see\n\n").append(getTechnicalInformation().toString());
        return sb.toString();
    }

}