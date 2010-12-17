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
 *    IBLR_ML.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.lazy;

import java.util.ArrayList;
import mulan.classifier.MultiLabelOutput;
import mulan.data.DataUtils;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.classifiers.functions.Logistic;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * 
 * <!-- globalinfo-start -->
 * This class is a re-implementation of the "IBLR-ML" and "IBLR-ML+" methods for the MULAN package.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Weiwei Cheng, Eyke Hullermeier (2009). Combining instance-based learning and logistic regression for multilabel classification . Machine Learning. 76(2-3):211-225.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Cheng2009,
 *    author = {Weiwei Cheng and Eyke Hullermeier},
 *    journal = {Machine Learning},
 *    number = {2-3},
 *    pages = {211-225},
 *    publisher = {Springer Netherlands},
 *    title = {Combining instance-based learning and logistic regression for multilabel classification },
 *    volume = {76},
 *    year = {2009},
 *    ISSN = {0885-6125}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * @author Weiwei Cheng
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * 
 */
public class IBLR_ML extends MultiLabelKNN {

    private static final long serialVersionUID = 1L;
    /**
     * For each label we create a corresponding binary classifier.
     */
    Classifier[] classifier;
    /**
     * By default, IBLR-ML is used. One can change to IBLR-ML+ with
     * {@link setAddFeatures}
     */
    boolean addFeatures = false;

    /**
     * Default constructor uses 10 NN
     */
    public IBLR_ML() {
        super(10);
    }

    /**
     * @param numNeighbors
     *            the number of nearest neighbors considered
     */
    public IBLR_ML(int numNeighbors) {
        super(numNeighbors);
    }

    /**
     * Returns a string describing classifier.
     *
     * @return a description suitable for displaying in a future
     *         explorer/experimenter gui
     */
    public String globalInfo() {

        return "This class is a re-implementation of the \"IBLR-ML\" and \"IBLR-ML+\" methods for the MULAN package." + "\n\n" + "For more information, see\n\n" + getTechnicalInformation().toString();
    }

    /**
     *
     * @param addFeatures
     */
    public void setAddFeatures(boolean addFeatures) {
        this.addFeatures = addFeatures;
    }

    @Override
    protected void buildInternal(MultiLabelInstances mltrain) throws Exception {
        super.buildInternal(mltrain);

        classifier = new Classifier[numLabels];

        /*
         * Create the new training data with label info as features.
         */
        Instances[] trainingDataForLabel = new Instances[numLabels];
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        if (addFeatures == true) {// create an ArrayList with numAttributes size
            for (int i = 1; i <= train.numAttributes(); i++) {
                attributes.add(new Attribute("Attr." + i));
            }
        } else {// create a FastVector with numLabels size
            for (int i = 1; i <= numLabels; i++) {
                attributes.add(new Attribute("Attr." + i));
            }
        }
        ArrayList<String> classlabel = new ArrayList<String>();
        classlabel.add("0");
        classlabel.add("1");
        attributes.add(new Attribute("Class", classlabel));
        for (int i = 0; i < trainingDataForLabel.length; i++) {
            trainingDataForLabel[i] = new Instances("DataForLabel" + (i + 1),
                    attributes, train.numInstances());
            trainingDataForLabel[i].setClassIndex(trainingDataForLabel[i].numAttributes() - 1);
        }

        for (int i = 0; i < train.numInstances(); i++) {

            Instances knn = new Instances(lnn.kNearestNeighbours(train.instance(i), numOfNeighbors));
            /*
             * Get the label confidence vector as the additional features.
             */
            double[] confidences = new double[numLabels];
            for (int j = 0; j < numLabels; j++) {
                // compute sum of counts for each label in KNN
                double count_for_label_j = 0;
                for (int k = 0; k < numOfNeighbors; k++) {
                    double value = Double.parseDouble(train.attribute(
                            labelIndices[j]).value(
                            (int) knn.instance(k).value(labelIndices[j])));
                    if (Utils.eq(value, 1.0)) {
                        count_for_label_j++;
                    }
                }
                confidences[j] = count_for_label_j / numOfNeighbors;
            }

            double[] attvalue = new double[numLabels + 1];

            if (addFeatures == true) {
                attvalue = new double[train.numAttributes() + 1];

                // Copy the original features
                for (int m = 0; m < featureIndices.length; m++) {
                    attvalue[m] = train.instance(i).value(featureIndices[m]);
                }
                // Copy the label confidences as additional features
                for (int m = 0; m < confidences.length; m++) {
                    attvalue[train.numAttributes() - numLabels + m] = confidences[m];
                }
            } else {
                // Copy the label confidences as features
                for (int m = 0; m < confidences.length; m++) {
                    attvalue[m] = confidences[m];
                }
            }

            // Add the class labels and finish the new training data
            for (int j = 0; j < numLabels; j++) {
                attvalue[attvalue.length - 1] = Double.parseDouble(train.attribute(labelIndices[j]).value(
                        (int) train.instance(i).value(labelIndices[j])));
                Instance newInst = DataUtils.createInstance(train.instance(i), 1, attvalue);
                newInst.setDataset(trainingDataForLabel[j]);
                if (attvalue[attvalue.length - 1] > 0.5) {
                    newInst.setClassValue("1");
                } else {
                    newInst.setClassValue("0");
                }
                trainingDataForLabel[j].add(newInst);
            }

        }

        // for every label create a corresponding classifier.
        for (int i = 0; i < numLabels; i++) {
            classifier[i] = new Logistic();
            classifier[i].buildClassifier(trainingDataForLabel[i]);
        }

    }

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {

        double[] conf_corrected = new double[numLabels];
        double[] confidences = new double[numLabels];

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

        double[] attvalue = new double[numLabels + 1];

        if (addFeatures == true) {
            attvalue = new double[instance.numAttributes() + 1];

            // Copy the original features
            for (int m = 0; m < featureIndices.length; m++) {
                attvalue[m] = instance.value(featureIndices[m]);
            }
            // Copy the label confidences as additional features
            for (int m = 0; m < confidences.length; m++) {
                attvalue[train.numAttributes() - numLabels + m] = confidences[m];
            }
        } else {
            // Copy the label confidences as additional features
            for (int m = 0; m < confidences.length; m++) {
                attvalue[m] = confidences[m];
            }
        }

        // Add the class labels and finish the new training data
        for (int j = 0; j < numLabels; j++) {
            attvalue[attvalue.length - 1] = instance.value(train.numAttributes() - numLabels + j);
            Instance newInst = DataUtils.createInstance(instance, 1, attvalue);
            conf_corrected[j] = classifier[j].distributionForInstance(newInst)[1];
        }

        MultiLabelOutput mlo = new MultiLabelOutput(conf_corrected, 0.5);
        return mlo;
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

        result = new TechnicalInformation(Type.ARTICLE);
        result.setValue(Field.AUTHOR, "Weiwei Cheng and Eyke Hullermeier");
        result.setValue(
                Field.TITLE,
                "Combining instance-based learning and logistic regression for multilabel classification ");
        result.setValue(Field.JOURNAL, "Machine Learning");
        result.setValue(Field.VOLUME, "76");
        result.setValue(Field.NUMBER, "2-3");
        result.setValue(Field.YEAR, "2009");
        result.setValue(Field.ISSN, "0885-6125");
        result.setValue(Field.PAGES, "211-225");
        result.setValue(Field.PUBLISHER, "Springer Netherlands");

        return result;
    }
}
