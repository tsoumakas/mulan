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
 *    Statistics.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
<!-- globalinfo-start -->
* Class for calculating statistics of a multi-label dataset. For more information, see<br>
* <br>
* Tsoumakas, Grigorios, Katakis, Ioannis, Vlahavas, Ioannis: Mining Multi-Label Data. In Maimon, Oded and Rokach, Lior, editors, Data Mining and Knowledge Discovery Handbook, 667-685, 2010.
* <br>
<!-- globalinfo-end -->
 * 
<!-- technical-bibtex-start -->
* BibTeX:
* <pre>
* &#64;incollection{Tsoumakas2010,
*    author = {Tsoumakas, Grigorios and Katakis, Ioannis and Vlahavas, Ioannis},
*    booktitle = {Data Mining and Knowledge Discovery Handbook},
*    edition = {2nd},
*    editor = {Maimon, Oded and Rokach, Lior},
*    pages = {667-685},
*    publisher = {Springer},
*    title = {Mining Multi-Label Data},
*    year = {2010}
* }
* </pre>
* <br>
<!-- technical-bibtex-end -->
 *
 *
 * @author Grigorios Tsoumakas 
 * @author Robert Friberg
 * @version 2012.02.06 
 */
public class Statistics implements Serializable, TechnicalInformationHandler {

    private static final long serialVersionUID = 1206845794397561633L;
    /** the number of instances */
    private int numInstances;
    /** the number of predictive attributes */
    private int numPredictors = 0;
    /** the number of nominal predictive attributes */
    private int numNominal = 0;
    /** the number of numeric attributes */
    private int numNumeric = 0;
    /** the number of labels */
    private int numLabels;
    /** the label density  */
    private double labelDensity;
    /** the label cardinality */
    private double labelCardinality;
    /** percentage of instances per label */
    private double[] examplesPerLabel;
    /** number of examples per cardinality, <br><br>
     *  note that this array has size equal to the number of elements plus one, <br>
     *  because the first element is the number of examples for cardinality=0  */
    private double[] cardinalityDistribution;
    /** labelsets and their frequency */
    private HashMap<LabelSet, Integer> labelsets;
    /** the array holding the phi correlations*/
    double[][] phi;

    /** 
     * returns the HashMap containing the distinct labelsets and their frequencies
     * 
     * @return HashMap with distinct labelsest and their frequencies
     */
    public HashMap<LabelSet, Integer> labelCombCount() {
        return labelsets;
    }

    /** 
     * This method calculates and prints a matrix with the coocurrences of <br>
     * pairs of labels
     *
     * @param mdata a multi-label data set
     * @return a matrix of co-occurences
     */
    public double[][] calculateCoocurrence(MultiLabelInstances mdata) {
        Instances data = mdata.getDataSet();
        int labels = mdata.getNumLabels();
        double[][] coocurrenceMatrix = new double[labels][labels];

        numPredictors = data.numAttributes() - labels;
        for (int k = 0; k < data.numInstances(); k++) {
            Instance temp = data.instance(k);
            for (int i = 0; i < labels; i++) {
                for (int j = 0; j < labels; j++) {
                    if (i >= j) {
                        continue;
                    }
                    if (temp.stringValue(numPredictors + i).equals("1") && temp.stringValue(numPredictors + j).equals("1")) {
                        coocurrenceMatrix[i][j]++;
                    }
                }
            }
        }

        for (int i = 0; i < labels; i++) {
            for (int j = 0; j < labels; j++) {
                System.out.print(coocurrenceMatrix[i][j] + "\t");
            }
            System.out.println();
        }

        return coocurrenceMatrix;
    }

    /** 
     * calculates various multilabel statistics, such as label cardinality, <br>
     * label density and the set of distinct labels along with their frequency
     * 
     * @param mlData a multi-label dataset
     */
    public void calculateStats(MultiLabelInstances mlData) {
        // initialize statistics
        Instances data = mlData.getDataSet();
        numLabels = mlData.getNumLabels();
        int[] labelIndices = mlData.getLabelIndices();
        int[] featureIndices = mlData.getFeatureIndices();
        numPredictors = featureIndices.length;

        labelCardinality = 0;
        numNominal = 0;
        numNumeric = 0;
        examplesPerLabel = new double[numLabels];
        cardinalityDistribution = new double[numLabels + 1];
        labelsets = new HashMap<LabelSet, Integer>();

        // gather statistics
        for (int i = 0; i < featureIndices.length; i++) {
            if (data.attribute(featureIndices[i]).isNominal()) {
                numNominal++;
            }
            if (data.attribute(featureIndices[i]).isNumeric()) {
                numNumeric++;
            }
        }

        numInstances = data.numInstances();
        for (int i = 0; i < numInstances; i++) {
            int exampleCardinality = 0;
            double[] dblLabels = new double[numLabels];
            for (int j = 0; j < numLabels; j++) {
                if (data.instance(i).stringValue(labelIndices[j]).equals("1")) {
                    dblLabels[j] = 1;
                    exampleCardinality++;
                    labelCardinality++;
                    examplesPerLabel[j]++;
                } else {
                    dblLabels[j] = 0;
                }
            }
            cardinalityDistribution[exampleCardinality]++;

            LabelSet labelSet = new LabelSet(dblLabels);
            if (labelsets.containsKey(labelSet)) {
                labelsets.put(labelSet, labelsets.get(labelSet) + 1);
            } else {
                labelsets.put(labelSet, 1);
            }
        }

        labelCardinality /= numInstances;
        labelDensity = labelCardinality / numLabels;
        for (int j = 0; j < numLabels; j++) {
            examplesPerLabel[j] /= numInstances;
        }
    }

    /**
     * Calculates phi correlation
     *
     * @param dataSet a multi-label dataset
     * @return a matrix containing phi correlations
     * @throws java.lang.Exception Potential exception thrown. To be handled in an upper level.
     */
    public double[][] calculatePhi(MultiLabelInstances dataSet) throws Exception {

        numLabels = dataSet.getNumLabels();

        /** the indices of the label attributes */
        int[] labelIndices;

        labelIndices = dataSet.getLabelIndices();
        numLabels = dataSet.getNumLabels();
        phi = new double[numLabels][numLabels];

        Remove remove = new Remove();
        remove.setInvertSelection(true);
        remove.setAttributeIndicesArray(labelIndices);
        remove.setInputFormat(dataSet.getDataSet());
        Instances result = Filter.useFilter(dataSet.getDataSet(), remove);
        result.setClassIndex(result.numAttributes() - 1);

        for (int i = 0; i < numLabels; i++) {

            int a[] = new int[numLabels];
            int b[] = new int[numLabels];
            int c[] = new int[numLabels];
            int d[] = new int[numLabels];
            double e[] = new double[numLabels];
            double f[] = new double[numLabels];
            double g[] = new double[numLabels];
            double h[] = new double[numLabels];

            for (int j = 0; j < result.numInstances(); j++) {
                for (int l = 0; l < numLabels; l++) {
                    if (result.instance(j).stringValue(i).equals("0")) {
                        if (result.instance(j).stringValue(l).equals("0")) {
                            a[l]++;
                        } else {
                            c[l]++;
                        }
                    } else {
                        if (result.instance(j).stringValue(l).equals("0")) {
                            b[l]++;
                        } else {
                            d[l]++;
                        }
                    }
                }
            }
            for (int l = 0; l < numLabels; l++) {
                e[l] = a[l] + b[l];
                f[l] = c[l] + d[l];
                g[l] = a[l] + c[l];
                h[l] = b[l] + d[l];

                double mult = e[l] * f[l] * g[l] * h[l];
                double denominator = Math.sqrt(mult);
                double nominator = a[l] * d[l] - b[l] * c[l];
                phi[i][l] = nominator / denominator;

            }
        }
        return phi;
    }

    /**
     * Prints out phi correlations
     */
    public void printPhiCorrelations() {
        String pattern = "0.00";
        DecimalFormat myFormatter = new DecimalFormat(pattern);

        for (int i = 0; i < numLabels; i++) {
            for (int j = 0; j < numLabels; j++) {
                System.out.print(myFormatter.format(phi[i][j]) + " ");
            }
            System.out.println("");
        }
    }

    /**
     * Calculates a histogram of phi correlations
     *
     * @return an array with phi correlations
     */
    public double[] getPhiHistogram() {
        double[] pairs = new double[numLabels * (numLabels - 1) / 2];
        int counter = 0;
        for (int i = 0; i < numLabels - 1; i++) {
            for (int j = i + 1; j < numLabels; j++) {
                pairs[counter] = phi[i][j];
                counter++;
            }
        }
        return pairs;
    }

    /**
     * returns the indices of the labels whose phi coefficient values lie
     * between -bound &lt;= phi &lt;= bound
     *
     * @param labelIndex the label index under examination
     * @param bound the bound which limits the phi coefficient values 
     * @return the indices of the labels whose phi coefficient values lie between -bound &lt;= phi &lt;= bound
     */
    public int[] uncorrelatedLabels(int labelIndex, double bound) {
        ArrayList<Integer> indiceslist = new ArrayList<Integer>();
        for (int i = 0; i < numLabels; i++) {
            if (Math.abs(phi[labelIndex][i]) <= bound) {
                indiceslist.add(i);
            }
        }
        int[] indices = new int[indiceslist.size()];
        for (int i = 0; i < indiceslist.size(); i++) {
            indices[i] = indiceslist.get(i);
        }

        return indices;
    }

    /**
     * Returns the indices of the labels that have the strongest phi correlation
     * with the label which is given as a parameter. The second parameter is
     * the number of labels that will be returned.
     *
     * @param labelIndex the label index under examination
     * @param k the number of labels to be returned
     * @return the indices of the k most correlated labels
     */
    public int[] topPhiCorrelatedLabels(int labelIndex, int k) {
        //create a new array containing the absolute values of the original array
        double[] absCorrelations = new double[numLabels];
        for (int i = 0; i < numLabels; i++) {
            absCorrelations[i] = Math.abs(phi[labelIndex][i]);
        }
        //sort the array of correlations
        int[] sorted = Utils.stableSort(absCorrelations);

        int[] topPhiCorrelated = new int[k + 1];
        //the k last values of the sorted array are the indices of the top k correlated labels
        for (int i = 0; i < k; i++) {
            topPhiCorrelated[i] = sorted[numLabels - 1 - i];
        }
        // one more for the class
        topPhiCorrelated[k] = numLabels;

        return topPhiCorrelated;
    }

    /**
     * This method prints data, useful for the visualization of Phi per dataset.
     * It prints int(1/step) + 1 pairs of values. The first value of each pair
     * is the phi value and the second is the average number of labels that
     * correlate to the rest of the labels with correlation higher than the
     * specified phi value;
     *
     * @param step
     *            the phi value increment step
     */
    public void printPhiDiagram(double step) {
        String pattern = "0.00";
        DecimalFormat myFormatter = new DecimalFormat(pattern);

        System.out.println("Phi      AvgCorrelated");
        double tempPhi = 0;
        while (tempPhi <= 1.001) {
            double avgCorrelated = 0;
            for (int i = 0; i < numLabels; i++) {
                int[] temp = uncorrelatedLabels(i, tempPhi);
                avgCorrelated += (numLabels - temp.length);
            }
            avgCorrelated /= numLabels;
            System.out.println(myFormatter.format(phi) + "     " + avgCorrelated);
            tempPhi += step;
        }
    }

    /** 
     * returns various multilabel statistics in textual representation 
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("Examples: ").append(numInstances).append("\n");
        sb.append("Predictors: ").append(numPredictors).append("\n");
        sb.append("--Nominal: ").append(numNominal).append("\n");
        sb.append("--Numeric: ").append(numNumeric).append("\n");

        sb.append("Labels: ").append(numLabels).append("\n");

        sb.append("\n");
        sb.append("Cardinality: ").append(labelCardinality).append("\n");
        sb.append("Density: ").append(labelDensity).append("\n");
        sb.append("Distinct Labelsets: ").append(labelsets.size()).append("\n");

        sb.append("\n");
        for (int j = 0; j < numLabels; j++) {
            sb.append("Percentage of examples with label ").append(j + 1).append(": ").append(examplesPerLabel[j]).append("\n");
        }

        sb.append("\n");
        for (int j = 0; j <= numLabels; j++) {
            sb.append("Examples of cardinality ").append(j).append(": ").append(cardinalityDistribution[j]).append("\n");
        }

        sb.append("\n");
        for (LabelSet set : labelsets.keySet()) {
            sb.append("Examples of combination ").append(set).append(": ").append(labelsets.get(set)).append("\n");
        }

        return sb.toString();
    }

    /** 
     * returns the prior probabilities of the labels
     * 
     * @return array of prior probabilities of labels
     */
    public double[] priors() {
        return examplesPerLabel;
    }

    /** 
     * returns the label cardinality of the dataset
     * 
     * @return label cardinality
     */
    public double cardinality() {
        return labelCardinality;
    }

    /** 
     * returns the label density of the dataset
     * 
     * @return label density
     */
    public double density() {
        return labelDensity;
    }

    /** 
     * returns a set with the distinct labelsets of the dataset
     * 
     * @return set of distinct labelsets
     */
    public Set<LabelSet> labelSets() {
        return labelsets.keySet();
    }

    /** 
     * returns the frequency of a labelset in the dataset
     * 
     * @param x a labelset
     * @return the frequency of the given labelset
     */
    public int labelFrequency(LabelSet x) {
        return labelsets.get(x);
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INCOLLECTION);
        result.setValue(Field.AUTHOR, "Tsoumakas, Grigorios and Katakis, Ioannis and Vlahavas, Ioannis");
        result.setValue(Field.TITLE, "Mining Multi-Label Data");
        result.setValue(Field.PAGES, "667-685");
        result.setValue(Field.BOOKTITLE, "Data Mining and Knowledge Discovery Handbook");
        result.setValue(Field.EDITOR, "Maimon, Oded and Rokach, Lior");
        result.setValue(Field.PUBLISHER, "Springer");
        result.setValue(Field.EDITION, "2nd");
        result.setValue(Field.YEAR, "2010");
        return result;
    }
    
    /**
     * Returns a string describing this class.
     *
     * @return a description suitable for displaying in a future gui
     */    public String globalInfo() {
        return "Class for calculating statistics of a multi-label dataset. "
                + "For more information, see\n\n"
                + getTechnicalInformation().toString();
    }

}