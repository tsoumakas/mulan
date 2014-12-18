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
 *    DataPair.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import mulan.core.ArgumentNullException;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class for representation of a data-pair instance. The data pair contains 
 * an input pattern and respected true or expected output pattern for the input pattern.
 * 
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public class DataPair {

    private final double[] input;
    private final double[] output;
    private boolean[] outputBoolean;

    /**
     * Creates a {@link DataPair} instance.
     * @param inputPattern the input pattern
     * @param trueOutput the true/expected output pattern for the input
     */
    public DataPair(final double[] inputPattern, final double[] trueOutput) {
        if (inputPattern == null) {
            throw new ArgumentNullException("inputPattern");
        }
        if (trueOutput == null) {
            throw new ArgumentNullException("trueOutput");
        }
        this.input = Arrays.copyOf(inputPattern, inputPattern.length);
        this.output = Arrays.copyOf(trueOutput, trueOutput.length);
    }

    /**
     * Gets the input pattern.
     * @return the input pattern
     */
    public double[] getInput() {
        return input;
    }

    /**
     * Gets the ideal/expected output pattern.
     * @return the output pattern
     */
    public double[] getOutput() {
        return output;
    }

    /**
     * Gets the ideal/expected output pattern as boolean values.
     * This is useful when output represents labels bipartition.
     * If output values in <code>double[]</code> are not in boolean representation,
     * then output of this method is might not be valid.
     * The computation is as follows:<br>
     * - if value is equal to 1, then output is <code>true</code> in boolean<br>
     * - if value is other than 1, then output is <code>false</code> in boolean
     *
     * @return the boolean representation of the output pattern
     */
    public boolean[] getOutputBoolean() {
        if (outputBoolean == null) {
            outputBoolean = new boolean[output.length];
            for (int i = 0; i < output.length; i++) {
                outputBoolean[i] = (output[i] == 1) ? true : false;
            }
        }

        return outputBoolean;
    }

    /**
     * Creates a {@link DataPair} representation for each {@link Instance} contained in
     * {@link MultiLabelInstances} data set. The {@link DataPair} is a light weight representation
     * of instance values (by double values), which is useful when iteration over the data and its
     * values.
     *
     * @param mlDataSet the {@link MultiLabelInstances} which content has to be
     * 			converted to list of {@link DataPair}
     * @param bipolarOutput indicates whether output values should be converted
     * 			to bipolar values, or left intact as binary
     * @return the list of data pairs
     */
    // TODO: this method should be in some kind of "data utils".
    public static List<DataPair> createDataPairs(MultiLabelInstances mlDataSet,
            boolean bipolarOutput) {

        Instances data = mlDataSet.getDataSet();
        int[] featureIndices = mlDataSet.getFeatureIndices();
        int[] labelIndices = mlDataSet.getLabelIndices();
        int numFeatures = featureIndices.length;
        int numLabels = mlDataSet.getNumLabels();

        int numInstances = data.numInstances();
        List<DataPair> dataPairs = new ArrayList<DataPair>(numInstances);
        for (int index = 0; index < numInstances; index++) {
            Instance instance = data.instance(index);
            double[] input = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) {
                int featureIndex = featureIndices[i];
                Attribute featureAttr = instance.attribute(featureIndex);
                // if attribute is binary, parse the string value ... it is expected to be '0' or '1'
                if (featureAttr.isNominal() && featureAttr.numValues() == 2) {
                    input[i] = Double.parseDouble(instance.stringValue(featureIndex));
                } // else :
                // a) the attribute is nominal with multiple values, use indexes as nominal values
                //    do not have to be numbers in general ... this is fall-back ... should be rare case
                // b) is numeric attribute
                else {
                    input[i] = instance.value(featureIndex);
                }
            }

            if (mlDataSet.hasMissingLabels(instance))
                continue;

            double[] output = new double[numLabels];
            for (int i = 0; i < numLabels; i++) {
                output[i] = Double.parseDouble(data.attribute(labelIndices[i]).value((int) instance.value(labelIndices[i])));
                if (bipolarOutput && output[i] == 0) {
                    output[i] = -1;
                }
            }

            dataPairs.add(new DataPair(input, output));
        }

        return dataPairs;
    }
}