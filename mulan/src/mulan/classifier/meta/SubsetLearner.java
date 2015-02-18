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

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.core.ArgumentNullException;
import mulan.data.ConditionalDependenceIdentifier;
import mulan.data.GreedyLabelClustering;
import mulan.data.LabelClustering;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 <!-- globalinfo-start -->
 * A class for learning a classifier according to disjoint label subsets: a multi-label learner (the Label Powerset by default) is applied to subsets with multiple labels and a single-label learner is applied to single label  subsets. The final classification prediction is  determined by combining labels predicted by all the learned models. Note: the class is not multi-thread safe. &lt;br&gt; &lt;br&gt; There is a mechanism for caching and reusing learned classification models. The caching mechanism is controlled by {&#64;link #useCache} parameter.<br>
 * <br>
 * For more information, see<br>
 * <br>
 * Lena Tenenboim, Lior Rokach,, Bracha Shapira: Multi-label Classification by Analyzing Labels Dependencies. In: , Bled, Slovenia, 117--132, 2009.<br>
 * <br>
 * Lena Tenenboim-Chekina, Lior Rokach,, Bracha Shapira: Identification of Label Dependencies for Multi-label Classification. In: , Haifa, Israel, 53--60, 2010.
 * <br>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{LenaTenenboim2009,
 *    address = {Bled, Slovenia},
 *    author = {Lena Tenenboim, Lior Rokach, and Bracha Shapira},
 *    pages = {117--132},
 *    title = {Multi-label Classification by Analyzing Labels Dependencies},
 *    volume = {Proc. ECML/PKDD 2009 Workshop on Learning from Multi-Label Data (MLD'09)},
 *    year = {2009}
 * }
 * 
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
 * @author Vasiloudis Theodoros
 * @version 30.11.2010
 */
public class SubsetLearner extends MultiLabelMetaLearner {

    /**
     * Arraylist containing the MultiLabelLearners that we will train and use to
     * make the predictions
     */
    private ArrayList<MultiLabelLearner> multiLabelLearners;
    /**
     * Arraylist containing the FilteredClassifiers that we will train and use
     * to make the predictions
     */
    private ArrayList<FilteredClassifier> singleLabelLearners;
    /**
     * Array containing the way the labels will be split
     */
    private int[][] splitOrder;
    /**
     * Array containing the indices of the labels we are going to remove
     */
    private int[][] absoluteIndicesToRemove;
    /**
     * Array containing the Remove objects used to remove the labels for each
     * split
     */
    private Remove[] remove;
    /**
     * Base single-label classifier that will be used for training and
     * predictions
     */
    protected Classifier baseSingleLabelClassifier;
    /**
     * indication for disabled caching mechanism
     */
    private boolean useCache = false;
    /**
     * The method used to cluster the labels
     */
    private LabelClustering clusterer = null;
    /**
     * HashMaps containing created models - caching mechanism is used, if
     * enabled by setting the useCache field to true, for GreedyLabelClustering
     * and EnsembleOfSubsetLearners methods run time optimization
     */
    private static HashMap<String, MultiLabelLearner> existingMultiLabelModels = new HashMap<String, MultiLabelLearner>();
    private static HashMap<String, FilteredClassifier> existingSingleLabelModels = new HashMap<String, FilteredClassifier>();
    private static HashMap<String, Remove> existingRemove = new HashMap<String, Remove>();

    /**
     * Default constructor
     */
    public SubsetLearner() {
        this(new GreedyLabelClustering(new BinaryRelevance(new J48()), new J48(), new ConditionalDependenceIdentifier(new J48())), new BinaryRelevance(new J48()), new J48());
    }
            
    
    /**
     * Initialize the SubsetLearner with labels subsets partitioning and single
     * label learner.
     * {@link mulan.classifier.transformation.LabelPowerset} method initialized
     * with the specified single label learner.will be used as multilabel
     * learner.
     *
     * @param labelsSubsets subsets of dependent labels
     * @param singleLabelClassifier method used for single label classification
     */
    public SubsetLearner(int[][] labelsSubsets, Classifier singleLabelClassifier) {
        super(new LabelPowerset(singleLabelClassifier));
        if (singleLabelClassifier == null) {
            throw new ArgumentNullException("singleLabelClassifier");
        }
        if (labelsSubsets == null) {
            throw new ArgumentNullException("labelsSubsets");
        }

        baseSingleLabelClassifier = singleLabelClassifier;
        splitOrder = labelsSubsets;
        absoluteIndicesToRemove = new int[splitOrder.length][];
    }

    /**
     * Initialize the SubsetLearner with labels set partitioning, multilabel and
     * single label learners.
     *
     * @param labelsSubsets subsets of dependent labels
     * @param multiLabelLearner method used for multilabel classification
     * @param singleLabelClassifier method used for single label classification
     */
    public SubsetLearner(int[][] labelsSubsets, MultiLabelLearner multiLabelLearner,
            Classifier singleLabelClassifier) {
        super(multiLabelLearner);

        if (singleLabelClassifier == null) {
            throw new ArgumentNullException("singleLabelClassifier");
        }
        if (labelsSubsets == null) {
            throw new ArgumentNullException("labelsSubsets");
        }

        baseSingleLabelClassifier = singleLabelClassifier;
        splitOrder = labelsSubsets;
        absoluteIndicesToRemove = new int[splitOrder.length][];
    }

    /**
     * Initialize the SubsetLearner with a label clustering method, multilabel
     * and single label learners.
     *
     * @param clusteringMethod the method used for clustering
     * @param multiLabelLearner method used for multilabel classification
     * @param singleLabelClassifier method used for single label classification
     */
    public SubsetLearner(LabelClustering clusteringMethod, MultiLabelLearner multiLabelLearner,
            Classifier singleLabelClassifier) {
        super(multiLabelLearner);

        if (clusteringMethod == null) {
            throw new ArgumentNullException("clusteringMethod");
        }
        if (singleLabelClassifier == null) {
            throw new ArgumentNullException("singleLabelClassifier");
        }

        baseSingleLabelClassifier = singleLabelClassifier;
        clusterer = clusteringMethod;

    }

    /**
     * Reset the label set partitioning.
     *
     * @param labelsSubsets - new label set partitioning
     */
    public void resetSubsets(int[][] labelsSubsets) {
        splitOrder = labelsSubsets;
        absoluteIndicesToRemove = new int[splitOrder.length][];
    }

    /**
     * We get the initial dataset through trainingSet. Then for each subset of
     * labels as specified by labelsSubsets we remove the unneeded labels and
     * train the classifiers using MultiLabelLearner for multi-label splits and
     * BinaryRelevance approach for single label splits. Each classification
     * model constructed on a certain training data for a certain labels subset
     * along with related Remove object is stored in HashMap and can be reused
     * when is needed next time.
     *
     * @param trainingSet The initial {@link mulan.data.MultiLabelInstances}
     * dataset
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
        if (clusterer != null) {
            splitOrder = clusterer.determineClusters(trainingSet);
            absoluteIndicesToRemove = new int[splitOrder.length][];
        }
        remove = new Remove[splitOrder.length];
        prepareIndicesToRemove();
        multiLabelLearners = new ArrayList<MultiLabelLearner>();
        // Create the lists which will contain the learners
        singleLabelLearners = new ArrayList<FilteredClassifier>();
        int countSingle = 0, countMulti = 0;
        for (int totalSplitNo = 0; totalSplitNo < splitOrder.length; totalSplitNo++) {
            // Ensure ascending order of label indexes in the subset
            Arrays.sort(splitOrder[totalSplitNo]);
            int foldHash = trainingSet.getDataSet().toString().hashCode();
            // create unique key of the trainingSet and the labels subset to be used for caching
            String modelKey = createKey(splitOrder[totalSplitNo], foldHash);
            if (splitOrder[totalSplitNo].length > 1) {
                buildMultiLabelModel(trainingSet, countMulti, totalSplitNo, modelKey);
                countMulti++;
            } else {
                buildSingleLabelModel(trainingSet, countSingle, totalSplitNo, modelKey);
                countSingle++;
            }
        }
    }

    /**
     * Get values into absoluteIndicesToRemove
     */
    private void prepareIndicesToRemove() {
        int numofSplits = splitOrder.length; // Number of sets the main is going to be split into
        for (int r = 0; r < splitOrder.length; r++) { // Initialization required to avoid NullPointer exception
            absoluteIndicesToRemove[r] = new int[numLabels - splitOrder[r].length];
        }
        boolean[][] Selected = new boolean[splitOrder.length][numLabels]; // Initialize an array containing which labels we want
        for (int i = 0; i < numofSplits; i++) { // Set true for the labels we need to keep
            for (int j = 0; j < splitOrder[i].length; j++) {
                Selected[i][splitOrder[i][j]] = true;
            }
        }
        for (int i = 0; i < numofSplits; i++) { // Get the labels we need to KEEP
            int k = 0;
            for (int j = 0; j < numLabels; j++) {
                if (!Selected[i][j]) {
                    absoluteIndicesToRemove[i][k] = labelIndices[j];
                    k++;
                }
            }
        }
    }

    /**
     * Construct multilabel model.
     *
     * @param trainingSet The initial {@link mulan.data.MultiLabelInstances}
     * dataset
     * @param countMulti the number of previous multilabel splits within the
     * label-set partition
     * @param totalSplitNo the total number of previous splits within the
     * label-set partition
     * @param modelKey the unique key of the trainingSet and the labels subset
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private void buildMultiLabelModel(MultiLabelInstances trainingSet, int countMulti,
            int totalSplitNo, String modelKey) throws Exception {
        if (useCache && existingMultiLabelModels.containsKey(modelKey)) { // try to get existing model from cache
            MultiLabelLearner model = existingMultiLabelModels.get(modelKey);
            resetRandomSeed(model); // reset random seed of the classifier to it's initial value,	 such that it will be
            //  equal to that if the classifier was just trained.
            multiLabelLearners.add(model.makeCopy());
            remove[totalSplitNo] = existingRemove.get(modelKey);
        } else {  // (there is no such model in cache) -> build it
            Instances trainSubset = trainingSet.getDataSet();
            remove[totalSplitNo] = new Remove();  // Remove the unneeded labels
            remove[totalSplitNo].setAttributeIndicesArray(absoluteIndicesToRemove[totalSplitNo]);
            remove[totalSplitNo].setInputFormat(trainSubset);
            remove[totalSplitNo].setInvertSelection(false);
            trainSubset = Filter.useFilter(trainSubset, remove[totalSplitNo]);
            multiLabelLearners.add(baseLearner.makeCopy()); // Reintegrate dataset and train learner
            multiLabelLearners.get(countMulti).build(
                    trainingSet.reintegrateModifiedDataSet(trainSubset));
            if (useCache) { // add trained model and related Remove object to cache
                existingMultiLabelModels.put(modelKey, multiLabelLearners.get(countMulti));
                existingRemove.put(modelKey, remove[totalSplitNo]);
            }
        }
    }

    /**
     * Construct single label model.
     *
     * @param trainingSet The initial {@link mulan.data.MultiLabelInstances}
     * dataset
     * @param countSingle the number of previous single-label splits within the
     * label-set partition
     * @param totalSplitNo the total number of previous splits within the
     * label-set partition
     * @param modelKey the unique key of the trainingSet and the labels subset
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    private void buildSingleLabelModel(MultiLabelInstances trainingSet, int countSingle,
            int totalSplitNo, String modelKey) throws Exception {
        if (useCache && existingSingleLabelModels.containsKey(modelKey)) {
            // if single-label model is in cache -> get it
            FilteredClassifier model = existingSingleLabelModels.get(modelKey);
            Classifier classifier = model.getClassifier();
            // reset random seed of the classifier to it's initial value, such that it will be equal to that if the classifier was just trained
            resetRandomSeed(classifier);
            singleLabelLearners.add(model);
            remove[totalSplitNo] = existingRemove.get(modelKey);
        } else { // the model is not in cache -> build the model and add it to cache
            singleLabelLearners.add(new FilteredClassifier()); // Initialize the FilteredClassifiers
            singleLabelLearners.get(countSingle).setClassifier(
                    AbstractClassifier.makeCopy(baseSingleLabelClassifier));
            Instances trainSubset = trainingSet.getDataSet();
            remove[totalSplitNo] = new Remove(); // Set the remove filter for the	 FilteredClassifiers
            remove[totalSplitNo].setAttributeIndicesArray(absoluteIndicesToRemove[totalSplitNo]);
            remove[totalSplitNo].setInputFormat(trainSubset);
            remove[totalSplitNo].setInvertSelection(false);
            singleLabelLearners.get(countSingle).setFilter(remove[totalSplitNo]);
            // Set the remaining label as the class index
            trainSubset.setClassIndex(labelIndices[splitOrder[totalSplitNo][0]]);
            singleLabelLearners.get(countSingle).buildClassifier(trainSubset); // train
            if (useCache) { // add trained model and related Remove object to cache
                existingSingleLabelModels.put(modelKey, singleLabelLearners.get(countSingle));
                existingRemove.put(modelKey, remove[totalSplitNo]);
            }
        }
    }

    /**
     * Concatenate all integers from an array with additional integer into a
     * single string.
     *
     * @param set an array representing labels subset
     * @param fold a hash code of the current training set
     * @return a string in the form: "_l1_l2_ ... ln_fold"
     */
    private String createKey(int[] set, int fold) {
        StringBuilder sb = new StringBuilder("_");
        for (int i : set) {
            sb.append(i);
            sb.append("_");
        }
        sb.append(fold);
        return sb.toString();
    }

    /**
     * Invokes the setSeed(1) or setRandomSeed(1) method of the supplied
     * object's Class, if such method exist.
     *
     * @param model which random seed should be reset.
     */
    public void resetRandomSeed(Object model) {
        Class aClass = model.getClass();
        Method method = null;
        try {
            method = aClass.getMethod("setSeed", int.class);
        } catch (NoSuchMethodException e) {
            try {
                method = aClass.getMethod("setRandomSeed", int.class);
            } catch (NoSuchMethodException e2) {
                debug("NoSuchMethodExceptions: " + e.getMessage() + " and " + e2.getMessage());
            }
        }
        try {
            if (method != null) {
                method.invoke(model, 1);
            }
        } catch (IllegalAccessException e) {
            debug("IllegalAccessException: " + e.getMessage());
        } catch (InvocationTargetException e) {
            debug("InvocationTargetException: " + e.getMessage());
        }
    }

    /**
     * Set random seed of all internal Learners to 1.
     */
    public void setSeed() {
        for (MultiLabelLearner learner : multiLabelLearners) {
            resetRandomSeed(learner);
        }
        for (FilteredClassifier learner : singleLabelLearners) {
            resetRandomSeed(learner);
        }
    }

    /**
     * We make a prediction using a different method depending on whether the
     * split has one or more labels
     *
     * @param instance the instance for classification prediction
     * @return the {@link mulan.classifier.MultiLabelOutput} classification
     * prediction for the instance
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        MultiLabelOutput[] MLO = new MultiLabelOutput[splitOrder.length];
        int singleSplitNo = 0, multiSplitNo = 0;
        boolean[][] BooleanSubsets = new boolean[splitOrder.length][];
        double[][] ConfidenceSubsets = new double[splitOrder.length][];
        for (int r = 0; r < splitOrder.length; r++) { // Initilization required to avoid NullPointer exception
            BooleanSubsets[r] = new boolean[splitOrder[r].length];
            ConfidenceSubsets[r] = new double[splitOrder[r].length];
        }
        boolean[] BipartitionOut = new boolean[numLabels];
        double[] ConfidenceOut = new double[numLabels];

        // Make a prediction for the instance in each separate dataset
        // The learners have been trained for each seperate dataset in buildInternal
        for (int i = 0; i < splitOrder.length; i++) {
            if (splitOrder[i].length == 1) { // Prediction for single label splits
                double distribution[];
                try {
                    distribution = singleLabelLearners.get(singleSplitNo).distributionForInstance(
                            instance);
                } catch (Exception e) {
                    System.out.println(e);
                    return null;
                }
                int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

                // Ensure correct predictions both for class values {0,1} and {1,0}
                Attribute classAttribute = singleLabelLearners.get(singleSplitNo).getFilter().getOutputFormat().classAttribute();
                BooleanSubsets[i][0] = (classAttribute.value(maxIndex).equals("1"));
                // The confidence of the label being equal to 1
                ConfidenceSubsets[i][0] = distribution[classAttribute.indexOfValue("1")];
                singleSplitNo++;
            } else { // Prediction for multi label splits
                remove[i].input(instance);
                remove[i].batchFinished();
                Instance newInstance = remove[i].output();
                MLO[multiSplitNo] = multiLabelLearners.get(multiSplitNo).makePrediction(newInstance);
                // Get each array of Bipartitions, confidences from each learner
                BooleanSubsets[i] = MLO[multiSplitNo].getBipartition();
                ConfidenceSubsets[i] = MLO[multiSplitNo].getConfidences();
                multiSplitNo++;
            }
        }
        // Concatenate the outputs while putting everything in its right place
        for (int i = 0; i < splitOrder.length; i++) {
            for (int j = 0; j < splitOrder[i].length; j++) {
                BipartitionOut[splitOrder[i][j]] = BooleanSubsets[i][j];
                ConfidenceOut[splitOrder[i][j]] = ConfidenceSubsets[i][j];
            }
        }
        return new MultiLabelOutput(BipartitionOut, ConfidenceOut);
    }

    /**
     * Sets whether cache mechanism will be used
     * 
     * @param useCache whether cache mechanism will be used
     */
    public void setUseCache(boolean useCache) {
        this.useCache = useCache;
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;
        result = new TechnicalInformation(TechnicalInformation.Type.INPROCEEDINGS);
        result.setValue(TechnicalInformation.Field.AUTHOR,
                "Lena Tenenboim, Lior Rokach, and Bracha Shapira");
        result.setValue(TechnicalInformation.Field.TITLE,
                "Multi-label Classification by Analyzing Labels Dependencies");
        result.setValue(TechnicalInformation.Field.VOLUME,
                "Proc. ECML/PKDD 2009 Workshop on Learning from Multi-Label Data (MLD'09)");
        result.setValue(TechnicalInformation.Field.YEAR, "2009");
        result.setValue(TechnicalInformation.Field.PAGES, "117--132");
        result.setValue(TechnicalInformation.Field.ADDRESS, "Bled, Slovenia");
        TechnicalInformation result2;
        result2 = new TechnicalInformation(TechnicalInformation.Type.INPROCEEDINGS);
        result2.setValue(TechnicalInformation.Field.AUTHOR,
                "Lena Tenenboim-Chekina, Lior Rokach, and Bracha Shapira");
        result2.setValue(TechnicalInformation.Field.TITLE,
                "Identification of Label Dependencies for Multi-label Classification");
        result2.setValue(TechnicalInformation.Field.VOLUME,
                "Proc. ICML 2010 Workshop on Learning from Multi-Label Data (MLD'10");
        result2.setValue(TechnicalInformation.Field.YEAR, "2010");
        result2.setValue(TechnicalInformation.Field.PAGES, "53--60");
        result2.setValue(TechnicalInformation.Field.ADDRESS, "Haifa, Israel");
        result.add(result2);
        return result;
    }

    /**
     * Returns a string representation of the model
     * 
     * @return a string representation of the model
     */
    public String getModel() {
        String out = "";
        for (int i = 0; i < multiLabelLearners.size(); i++) {
            out += ((LabelPowerset) multiLabelLearners.get(i)).getBaseClassifier().toString();
        }
        return out;
    }
    
    public String globalInfo() {
        StringBuilder sb = new StringBuilder();
        sb.append("A class for learning a classifier according to disjoint ");
        sb.append("label subsets: a multi-label learner (the Label Powerset ");
        sb.append("by default) is applied to subsets with multiple labels and");
        sb.append(" a single-label learner is applied to single label ");
        sb.append(" subsets. The final classification prediction is ");
        sb.append(" determined by combining labels predicted by all the ");
        sb.append("learned models. Note: the class is not multi-thread safe. "); 
        sb.append("<br> <br> There is a mechanism for caching and reusing ");
        sb.append("learned classification models. The caching mechanism is ");
        sb.append("controlled by {@link #useCache} parameter.\n\nFor more ");
        sb.append("information, see\n\n");
        sb.append(getTechnicalInformation().toString());
        return sb.toString();
    }
}