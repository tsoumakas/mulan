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
 *    MultiLabelStacking.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.transformation;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.data.Statistics;
import mulan.data.DataUtils;
import mulan.transformations.BinaryRelevanceTransformation;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.Ranker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * <!-- globalinfo-start -->
 * This class is an implementation of the (BR)^2 or Multi-Label stacking method.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Grigorios Tsoumakas, Anastasios Dimou, Eleftherios Spyromitros, Vasileios Mezaris, Ioannis Kompatsiaris, Ioannis Vlahavas: Correlation-Based Pruning of Stacked Binary Relevance Models for Multi-Label Learning. In: Proc. ECML/PKDD 2009 Workshop on Learning from Multi-Label Data (MLD'09), 101-116, 2009.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inproceedings{GrigoriosTsoumakas2009,
 *    author = {Grigorios Tsoumakas, Anastasios Dimou, Eleftherios Spyromitros, Vasileios Mezaris, Ioannis Kompatsiaris, Ioannis Vlahavas},
 *    booktitle = {Proc. ECML/PKDD 2009 Workshop on Learning from Multi-Label Data (MLD'09)},
 *    pages = {101-116},
 *    title = {Correlation-Based Pruning of Stacked Binary Relevance Models for Multi-Label Learning},
 *    year = {2009},
 *    location = {Bled, Slovenia}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 *
 */
public class MultiLabelStacking extends TransformationBasedMultiLabelLearner
        implements Serializable {

    private static final long serialVersionUID = 1L;
    /** the type of the classifier used in the meta-level */
    private Classifier metaClassifier;
    /** the BR transformed datasets of the original dataset */
    private Instances[] baseLevelData;
    /** the BR transformed datasets of the meta dataset */
    private Instances[] metaLevelData;
    /** the ensemble of BR classifiers of the original dataset */
    private Classifier[] baseLevelEnsemble;
    /** the ensemble of BR classifiers of the meta dataset */
    private Classifier[] metaLevelEnsemble;
    /** the ensemble of pruned BR classifiers of the meta dataset */
    private FilteredClassifier[] metaLevelFilteredEnsemble;
    /** the number of folds used in the first level */
    private int numFolds;
    /** the training instances */
    protected Instances train;
    /**
     * a table holding the predictions of the first level classifiers for each
     * class-label of every instance
     */
    private double[][] baseLevelPredictions;
    /** whether to normalize baseLevelPredictions or not. */
    private boolean normalize;
    /**
     * a table holding the maximum probability of each label according to the
     * predictions of the base level classifiers
     */
    private double maxProb[];
    /**
     * a table holding the minimum probability of each label according to the
     * predictions of the base level classifiers
     */
    private double minProb[];
    /** whether to include the original attributes in the meta-level */
    private boolean includeAttrs;
    /** defines the percentage of labels used in the meta-level */
    private double metaPercentage;
    /**
     * The number of labels that will be used for training the meta-level
     * classifiers. The value is derived by metaPercentage and used only
     * internally
     */
    private int topkCorrelated;
    /**
     * A table holding the attributes of the most correlated labels for each
     * label.
     */
    private int[][] selectedAttributes;
    /**
     * The attribute selection evaluator used for pruning the meta-level
     * attributes.
     */
    private ASEvaluation eval;
    /**
     * Class implementing the brute force search algorithm for nearest neighbor
     * search. Used only in case of a kNN baseClassifier
     */
    private LinearNNSearch lnn = null;
    /**
     * Whether base and meta level are going to be built separately.
     * If true then the buildInternal method doesn't build anything.
     */
    private boolean partialBuild;

    /*
     * private BRkNN brknn;
     */
    /**
     * Returns a string describing classifier.
     *
     * @return a description suitable for displaying in a future
     *         explorer/experimenter gui
     */
    public String globalInfo() {

        return "This class is an implementation of the (BR)^2 or Multi-Label stacking method." + "\n\n" + "For more information, see\n\n" + getTechnicalInformation().toString();
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(
                Field.AUTHOR,
                "Grigorios Tsoumakas, Anastasios Dimou, Eleftherios Spyromitros, Vasileios Mezaris, Ioannis Kompatsiaris, Ioannis Vlahavas");
        result.setValue(
                Field.TITLE,
                "Correlation-Based Pruning of Stacked Binary Relevance Models for Multi-Label Learning");
        result.setValue(Field.BOOKTITLE,
                "Proc. ECML/PKDD 2009 Workshop on Learning from Multi-Label Data (MLD'09)");
        result.setValue(Field.YEAR, "2009");
        result.setValue(Field.PAGES, "101-116");
        result.setValue(Field.LOCATION, "Bled, Slovenia");

        return result;
    }

    /**
     * An empty constructor
     */
    public MultiLabelStacking() {
    }

    /**
     * A constructor with 2 arguments
     *
     * @param baseClassifier the classifier used in the base-level
     * @param metaClassifier the classifier used in the meta-level
     * @throws Exception
     */
    public MultiLabelStacking(Classifier baseClassifier,
            Classifier metaClassifier) throws Exception {
        super(baseClassifier);
        this.metaClassifier = metaClassifier;
        this.numFolds = 10; // 10 folds by default
        metaPercentage = 1.0; // use 100% of the labels in the meta-level
        eval = null; // no feature selection by default
        normalize = false; // no normalization performed
        includeAttrs = false; // original attributes are not included
        partialBuild = false;
    }

    @Override
    protected void buildInternal(MultiLabelInstances dataSet) throws Exception {
        if (partialBuild) { // build base/meta level will be called separately
            return;
        }

        if (baseClassifier instanceof IBk) {
            buildBaseLevelKNN(dataSet);
        } else {
            buildBaseLevel(dataSet);
        }

        initializeMetaLevel(dataSet, metaClassifier, includeAttrs,
                metaPercentage, eval);

        buildMetaLevel();
    }

    /**
     * Initializes all the parameters used in the meta-level.
     * Calculates the correlated labels if meta-level pruning is applied.
     *
     * @param dataSet
     * @param metaClassifier
     * @param includeAttrs
     * @param metaPercentage
     * @param eval
     * @throws Exception
     */
    public void initializeMetaLevel(MultiLabelInstances dataSet,
            Classifier metaClassifier, boolean includeAttrs,
            double metaPercentage, ASEvaluation eval) throws Exception {
        this.metaClassifier = metaClassifier;
        metaLevelEnsemble = AbstractClassifier.makeCopies(metaClassifier, numLabels);
        metaLevelData = new Instances[numLabels];
        metaLevelFilteredEnsemble = new FilteredClassifier[numLabels];
        this.includeAttrs = includeAttrs;
        // calculate the number of correlated labels that corresponds to the
        // given percentage
        topkCorrelated = (int) Math.floor(metaPercentage * numLabels);
        if (topkCorrelated < 1) {
            debug("Too small percentage, selecting k=1");
            topkCorrelated = 1;
        }
        if (topkCorrelated < numLabels) {// pruning should be applied
            selectedAttributes = new int[numLabels][];
            if (eval == null) {// calculate the PhiCoefficient
                Statistics phi = new Statistics();
                phi.calculatePhi(dataSet);
                for (int i = 0; i < numLabels; i++) {
                    selectedAttributes[i] = phi.topPhiCorrelatedLabels(i,
                            topkCorrelated);
                }
            } else {// apply feature selection
                AttributeSelection attsel = new AttributeSelection();
                Ranker rankingMethod = new Ranker();
                rankingMethod.setNumToSelect(topkCorrelated);
                attsel.setEvaluator(eval);
                attsel.setSearch(rankingMethod);
                // create a dataset consisting of all the classes of each
                // instance plus the class we want to select attributes from
                for (int i = 0; i < numLabels; i++) {
                    ArrayList<Attribute> attributes = new ArrayList<Attribute>();

                    for (int j = 0; j < numLabels; j++) {
                        attributes.add(train.attribute(labelIndices[j]));
                    }
                    attributes.add(train.attribute(labelIndices[i]).copy("meta"));

                    Instances iporesult = new Instances("Meta format",
                            attributes, 0);
                    iporesult.setClassIndex(numLabels);
                    for (int k = 0; k < train.numInstances(); k++) {
                        double[] values = new double[numLabels + 1];
                        for (int m = 0; m < numLabels; m++) {
                            values[m] = Double.parseDouble(train.attribute(
									labelIndices[m]).value(
											(int) train.instance(k).value(
													labelIndices[m])));
                        }
                        values[numLabels] = Double
						.parseDouble(train.attribute(labelIndices[i])
								.value(
										(int) train.instance(k).value(
												labelIndices[i])));
                        Instance metaInstance = DataUtils.createInstance(train.instance(k), 1, values);
                        metaInstance.setDataset(iporesult);
                        iporesult.add(metaInstance);
                    }
                    attsel.SelectAttributes(iporesult);
                    selectedAttributes[i] = attsel.selectedAttributes();
                    iporesult.delete();
                }
            }
        }
    }

    /**
     * Builds the base-level classifiers.
     * Their predictions are gathered in the baseLevelPredictions member
     * @param trainingSet 
     * @throws Exception
     */
    public void buildBaseLevel(MultiLabelInstances trainingSet) throws Exception {
        train = new Instances(trainingSet.getDataSet());
        baseLevelData = new Instances[numLabels];
        baseLevelEnsemble = AbstractClassifier.makeCopies(baseClassifier, numLabels);
        if (normalize) {
            maxProb = new double[numLabels];
            minProb = new double[numLabels];
            Arrays.fill(minProb, 1);
        }
        // initialize the table holding the predictions of the first level
        // classifiers for each label for every instance of the training set
        baseLevelPredictions = new double[train.numInstances()][numLabels];

        for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
            debug("Label: " + labelIndex);
            // transform the dataset according to the BR method
            baseLevelData[labelIndex] = BinaryRelevanceTransformation.transformInstances(train, labelIndices,
                    labelIndices[labelIndex]);
            // attach indexes in order to keep track of the original positions
            baseLevelData[labelIndex] = new Instances(
                    attachIndexes(baseLevelData[labelIndex]));
            // prepare the transformed dataset for stratified x-fold cv
            Random random = new Random(1);
            baseLevelData[labelIndex].randomize(random);
            baseLevelData[labelIndex].stratify(numFolds);
            debug("Creating meta-data");
            for (int j = 0; j < numFolds; j++) {
                debug("Label=" + labelIndex + ", Fold=" + j);
                Instances subtrain = baseLevelData[labelIndex].trainCV(
                        numFolds, j, random);
                // create a filtered meta classifier, used to ignore
                // the index attribute in the build process
                // perform stratified x-fold cv and get predictions
                // for each class for every instance
                FilteredClassifier fil = new FilteredClassifier();
                fil.setClassifier(baseLevelEnsemble[labelIndex]);
                Remove remove = new Remove();
                remove.setAttributeIndices("first");
                remove.setInputFormat(subtrain);
                fil.setFilter(remove);
                fil.buildClassifier(subtrain);

                // Classify test instance
                Instances subtest = baseLevelData[labelIndex].testCV(numFolds,
                        j);
                for (int i = 0; i < subtest.numInstances(); i++) {
                    double distribution[] = new double[2];
                    distribution = fil.distributionForInstance(subtest.instance(i));
                    // Ensure correct predictions both for class values {0,1}
                    // and {1,0}
                    Attribute classAttribute = baseLevelData[labelIndex].classAttribute();
                    baseLevelPredictions[(int) subtest.instance(i).value(0)][labelIndex] = distribution[classAttribute.indexOfValue("1")];
                    if (normalize) {
                        if (distribution[classAttribute.indexOfValue("1")] > maxProb[labelIndex]) {
                            maxProb[labelIndex] = distribution[classAttribute.indexOfValue("1")];
                        }
                        if (distribution[classAttribute.indexOfValue("1")] < minProb[labelIndex]) {
                            minProb[labelIndex] = distribution[classAttribute.indexOfValue("1")];
                        }
                    }
                }
            }
            // now we can detach the indexes from the first level datasets
            baseLevelData[labelIndex] = detachIndexes(baseLevelData[labelIndex]);

            debug("Building base classifier on full data");
            // build base classifier on the full training data
            baseLevelEnsemble[labelIndex].buildClassifier(baseLevelData[labelIndex]);
            baseLevelData[labelIndex].delete();
        }

        if (normalize) {
            normalizePredictions();
        }

    }

    /**
     * Builds the ensemble of meta-level classifiers.
     *
     * @throws Exception
     */
    public void buildMetaLevel() throws Exception {
        debug("Building the ensemle of the meta level classifiers");

        for (int i = 0; i < numLabels; i++) { // creating meta-level data new
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();

            if (includeAttrs) {// create an ArrayList with numAttributes size
                for (int j = 0; j < train.numAttributes(); j++) {
                    attributes.add(train.attribute(j));
                }
            } else {// create a FastVector with numLabels size
                for (int j = 0; j < numLabels; j++) {
                    attributes.add(train.attribute(labelIndices[j]));
                }
            }
            attributes.add(train.attribute(labelIndices[i]).copy("meta"));

            metaLevelData[i] = new Instances("Meta format", attributes, 0);
            metaLevelData[i].setClassIndex(metaLevelData[i].numAttributes() - 1);

            // add the meta instances new
            for (int l = 0; l < train.numInstances(); l++) {
                double[] values = new double[metaLevelData[i].numAttributes()];
                if (includeAttrs) {
                    // Copy the original features
                    for (int m = 0; m < featureIndices.length; m++) {
                        values[m] = train.instance(l).value(featureIndices[m]);
                    }
                    // Copy the label confidences as additional features
                    for (int m = 0; m < numLabels; m++) {
                        values[train.numAttributes() - numLabels + m] = baseLevelPredictions[l][m];
                    }
                } else {
                    for (int m = 0; m < numLabels; m++) {
                        values[m] = baseLevelPredictions[l][m];
                    }
                }

                values[values.length - 1] = Double.parseDouble(train.attribute(
						labelIndices[i]).value(
								(int) train.instance(l).value(labelIndices[i])));
                Instance metaInstance = DataUtils.createInstance(train.instance(l), 1, values);
                metaInstance.setDataset(metaLevelData[i]);
                if (values[values.length - 1] > 0.5) {
                	metaInstance.setClassValue("1");
                } else {
                	metaInstance.setClassValue("0");
                }
                metaLevelData[i].add(metaInstance);
            }

            // We utilize a filtered classifier to prune uncorrelated labels
            metaLevelFilteredEnsemble[i] = new FilteredClassifier();
            metaLevelFilteredEnsemble[i].setClassifier(metaLevelEnsemble[i]);

            Remove remove = new Remove();

            if (topkCorrelated < numLabels) {
                remove.setAttributeIndicesArray(selectedAttributes[i]);
            } else {
                remove.setAttributeIndices("first-last");
            }

            remove.setInvertSelection(true);
            remove.setInputFormat(metaLevelData[i]);
            metaLevelFilteredEnsemble[i].setFilter(remove);

            debug("Building classifier for meta training set" + i);
            metaLevelFilteredEnsemble[i].buildClassifier(metaLevelData[i]);
            metaLevelData[i].delete();
        }
    }

    /**
     * Used only in case of a kNN base classifier.
     *
     * @param trainingSet
     * @throws Exception
     */
    public void buildBaseLevelKNN(MultiLabelInstances trainingSet)
            throws Exception {
        train = new Instances(trainingSet.getDataSet());
        EuclideanDistance dfunc = new EuclideanDistance();
        dfunc.setDontNormalize(false);

        // label attributes don't influence distance estimation
        String labelIndicesString = "";
        for (int i = 0; i < numLabels - 1; i++) {
            labelIndicesString += (labelIndices[i] + 1) + ",";
        }
        labelIndicesString += (labelIndices[numLabels - 1] + 1);
        dfunc.setAttributeIndices(labelIndicesString);
        dfunc.setInvertSelection(true);

        lnn = new LinearNNSearch();
        lnn.setSkipIdentical(true);
        lnn.setDistanceFunction(dfunc);
        lnn.setInstances(train);
        lnn.setMeasurePerformance(false);
        // initialize the table holding the predictions of the first level
        // classifiers for each label for every instance of the training set
        baseLevelPredictions = new double[train.numInstances()][numLabels];
        int numOfNeighbors = ((IBk) baseClassifier).getKNN();

        /*
         * /old way using brknn
         * brknn = new BRkNN(numOfNeighbors);
         * brknn.setDebug(true); brknn.build(trainingSet); for (int i = 0; i <
         * train.numInstances(); i++) { MultiLabelOutput prediction =
         * brknn.makePrediction(train.instance(i)); baseLevelPredictions[i] =
         * prediction.getConfidences(); }
         */

        // new way
        for (int i = 0; i < train.numInstances(); i++) {
            Instances knn = new Instances(lnn.kNearestNeighbours(train.instance(i), numOfNeighbors));

            // Get the label confidence vector as the additional features.

            for (int j = 0; j < numLabels; j++) {
                // compute sum of counts for each label in KNN
                double count_for_label_j = 0;
                for (int k = 0; k < numOfNeighbors; k++) {
                    String value = train.attribute(labelIndices[j]).value((int) knn.instance(k).value(labelIndices[j]));
                    if (value.equals("1")) {
                        count_for_label_j++;
                    }
                }
                baseLevelPredictions[i][j] = count_for_label_j / numOfNeighbors;
            }
        }

    }

    /**
     * Normalizes the predictions of the base-level classifiers
     */
    private void normalizePredictions() {
        for (int i = 0; i < baseLevelPredictions.length; i++) {
            for (int j = 0; j < numLabels; j++) {
                baseLevelPredictions[i][j] = baseLevelPredictions[i][j] - minProb[j] / maxProb[j] - minProb[j];
            }
        }
    }

    @Override
    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        // the confidences given as final output
        double[] metaconfidences = new double[numLabels];
        // the confidences produced by the first level ensemble of classfiers
        double[] confidences = new double[numLabels];

        if (!(baseClassifier instanceof IBk)) {
            // getting the confidences for each label
            for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
                Instance newInstance = BinaryRelevanceTransformation.transformInstance(instance, labelIndices,
                        labelIndices[labelIndex]);
                newInstance.setDataset(baseLevelData[labelIndex]);

                double distribution[] = new double[2];
                distribution = baseLevelEnsemble[labelIndex].distributionForInstance(newInstance);

                // Ensure correct predictions both for class values {0,1} and
                // {1,0}
                Attribute classAttribute = baseLevelData[labelIndex].classAttribute();
                // The confidence of the label being equal to 1
                confidences[labelIndex] = distribution[classAttribute.indexOfValue("1")];
            }
        } else {
            // old way using brknn
            // MultiLabelOutput prediction = brknn.makePrediction(instance);
            // confidences = prediction.getConfidences();

            // new way
            int numOfNeighbors = ((IBk) baseClassifier).getKNN();
            Instances knn = new Instances(lnn.kNearestNeighbours(instance,
                    numOfNeighbors));

            /*
             * Get the label confidence vector.
             */
            for (int i = 0; i < numLabels; i++) {
                // compute sum of counts for each label in KNN
                double count_for_label_i = 0;
                for (int k = 0; k < numOfNeighbors; k++) {
                    double value = Double.parseDouble(train.attribute(
                            labelIndices[i]).value(
                            (int) knn.instance(k).value(labelIndices[i])));
                    if (Utils.eq(value, 1.0)) {
                        count_for_label_i++;
                    }
                }

                confidences[i] = count_for_label_i / numOfNeighbors;

            }
        }
        // System.out.println(Utils.arrayToString(confidences));
		/* creation of the meta-instance with the appropriate values */
        double[] values = new double[numLabels + 1];

        if (includeAttrs) {
            values = new double[instance.numAttributes() + 1];
            // Copy the original features
            for (int m = 0; m < featureIndices.length; m++) {
                values[m] = instance.value(featureIndices[m]);
            }
            // Copy the label confidences as additional features
            for (int m = 0; m < confidences.length; m++) {
                values[instance.numAttributes() - numLabels + m] = confidences[m];
            }
        } else {
            for (int m = 0; m < confidences.length; m++) {
                values[m] = confidences[m];
            }
        }

        /* application of the meta-level ensemble to the metaInstance */
        for (int labelIndex = 0; labelIndex < numLabels; labelIndex++) {
            // values[values.length - 1] =
            // instance.value(instance.numAttributes() - numLabels +
            // labelIndex);
            values[values.length - 1] = 0;
            Instance newmetaInstance = DataUtils.createInstance(instance, 1, values);

            double distribution[] = new double[2];
            try {
                distribution = metaLevelFilteredEnsemble[labelIndex].distributionForInstance(newmetaInstance);
            } catch (Exception e) {
                System.out.println(e);
                return null;
            }
            int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

            // Ensure correct predictions both for class values {0,1} and {1,0}
            Attribute classAttribute = metaLevelData[labelIndex].classAttribute();
            bipartition[labelIndex] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

            // The confidence of the label being equal to 1
            metaconfidences[labelIndex] = distribution[classAttribute.indexOfValue("1")];
        }

        MultiLabelOutput mlo = new MultiLabelOutput(bipartition,
                metaconfidences);
        return mlo;
    }

    /**
     * Attaches an index attribute at the beginning of each instance
     *
     * @param original
     * @return
     */
    protected Instances attachIndexes(Instances original) {

        ArrayList<Attribute> attributes = new ArrayList<Attribute>(original.numAttributes() + 1);

        for (int i = 0; i < original.numAttributes(); i++) {
            attributes.add(original.attribute(i));
        }
        // Add attribute for holding the index at the beginning.
        attributes.add(0, new Attribute("Index"));
        Instances transformed = new Instances("Meta format", attributes, 0);
        for (int i = 0; i < original.numInstances(); i++) {
            Instance newInstance;
            newInstance = (Instance) original.instance(i).copy();
            newInstance.setDataset(null);
            newInstance.insertAttributeAt(0);
            newInstance.setValue(0, i);

            transformed.add(newInstance);
        }

        transformed.setClassIndex(original.classIndex() + 1);
        return transformed;
    }

    /**
     * Detaches the index attribute from the beginning of each instance
     *
     * @param original
     * @return
     * @throws Exception
     */
    protected Instances detachIndexes(Instances original) throws Exception {

        Remove remove = new Remove();
        remove.setAttributeIndices("first");
        remove.setInputFormat(original);
        Instances result = Filter.useFilter(original, remove);

        return result;

    }

    /**
     * Saves a {@link MultiLabelStacking} object in a file
     *
     * @param filename
     */
    public void saveObject(String filename) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(
                    new FileOutputStream(filename));
            out.writeObject(this);
        } catch (IOException ex) {
            Logger.getLogger(MultiLabelStacking.class.getName()).log(
                    Level.SEVERE, null, ex);
        }
    }

    /**
     * Sets the value of normalize
     *
     * @param normalize
     */
    public void setNormalize(boolean normalize) {
        this.normalize = normalize;
    }

    /**
     * Sets the value of includeAttrs
     *
     * @param includeAttrs
     */
    public void setIncludeAttrs(boolean includeAttrs) {
        this.includeAttrs = includeAttrs;
    }

    /**
     * Sets the value of metaPercentage
     *
     * @param metaPercentage
     */
    public void setMetaPercentage(double metaPercentage) {
        this.metaPercentage = metaPercentage;
    }

    /**
     * Sets the attribute selection evaluation class
     *
     * @param eval
     */
    public void setEval(ASEvaluation eval) {
        this.eval = eval;
    }

    /**
     * Sets the type of the meta classifier and initializes the ensemble
     *
     * @param metaClassifier
     * @throws Exception
     */
    public void setMetaAlgorithm(Classifier metaClassifier) throws Exception {
        this.metaClassifier = metaClassifier;
        metaLevelEnsemble = AbstractClassifier.makeCopies(metaClassifier, numLabels);
    }

    /**
     * sets the value for partialBuild
     * @param partialBuild 
     */
    public void setPartialBuild(boolean partialBuild) {
        this.partialBuild = partialBuild;
    }

	public int getTopkCorrelated() {
		return topkCorrelated;
	}
}
