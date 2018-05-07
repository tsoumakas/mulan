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
 *    ThresholdFunction.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural;

import java.io.Serializable;
import java.util.Arrays;
import weka.core.Utils;
import weka.core.matrix.Matrix;

/**
 * Implementation of a threshold function. 
 *  
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public class ThresholdFunction implements Serializable {

    /** Default serial version UID for serialization*/
    private static final long serialVersionUID = 5347411552628371402L;
    private double[] parameters;

    /**
     * Creates a new instance of {@link ThresholdFunction} and
     * builds the function based on input parameters.
     *
     * @param idealLabels the ideal output for each input patterns, which a model should output
     * @param modelOutLabels the real output of a model for each input pattern
     * @throws IllegalArgumentException if dimensions of input arrays does not match
     * @see ThresholdFunction#build(double[][], double[][])
     */
    public ThresholdFunction(final double[][] idealLabels, final double[][] modelOutLabels) {
        this.build(idealLabels, modelOutLabels);
    }

    /**
     * Computes a threshold value, based on learned parameters, for given labels confidences.
     *
     * @param labelsConfidences the labels confidences
     * @return the threshold value
     * @throws IllegalArgumentException if the dimension of labels confidences does not match
     * 		   							the dimension of learned parameters of threshold function.
     */
    public double computeThreshold(final double[] labelsConfidences) {

        int expectedDim = parameters.length - 1;
        if (labelsConfidences.length != expectedDim) {
            throw new IllegalArgumentException("The array of label confidences has wrong dimension." +
                    "The function expect parameters of length : " + expectedDim);
        }

        double threshold = 0;
        for (int index = 0; index < expectedDim; index++) {
            threshold += labelsConfidences[index] * parameters[index];
        }
        threshold += parameters[expectedDim];

        return threshold;
    }

    /**
     * Build a threshold function for based on input data.
     * The threshold function is build for a particular model.
     *
     * @param idealLabels the ideal output for each input patterns, which a model should output.
     * 					  First index is expected to be number of examples and second is the label index.
     * @param modelOutLabels the real output of a model for each input pattern.
     * 						 First index is expected to be number of examples and second is the label index.
     * @throws IllegalArgumentException if dimensions of input arrays does not match
     */
    public void build(final double[][] idealLabels, final double[][] modelOutLabels) {

        if (idealLabels == null || modelOutLabels == null) {
            throw new IllegalArgumentException("Non of the input parameters can be null.");
        }

        int numExamples = idealLabels.length;
        int numLabels = idealLabels[0].length;

        if (modelOutLabels.length != numExamples ||
                modelOutLabels[0].length != numLabels) {
            throw new IllegalArgumentException("Matrix dimensions of input parameters does not agree.");
        }

        double[] thresholds = new double[numExamples];
        double[] isLabelModelOuts = new double[numLabels];
        double[] isNotLabelModelOuts = new double[numLabels];
        for (int example = 0; example < numExamples; example++) {
            Arrays.fill(isLabelModelOuts, Double.MAX_VALUE);
            Arrays.fill(isNotLabelModelOuts, -Double.MAX_VALUE);
            for (int label = 0; label < numLabels; label++) {
                if (idealLabels[example][label] == 1) {
                    isLabelModelOuts[label] = modelOutLabels[example][label];
                } else {
                    isNotLabelModelOuts[label] = modelOutLabels[example][label];
                }
            }
            double isLabelMin = isLabelModelOuts[Utils.minIndex(isLabelModelOuts)];
            double isNotLabelMax = isNotLabelModelOuts[Utils.maxIndex(isNotLabelModelOuts)];

            // check if we have unique minimum ...
            // if not take center of the segment ... if it is a segment
            if (isLabelMin != isNotLabelMax) {
                // check marginal cases -> all labels are in or none of them
                if (isLabelMin == Double.MAX_VALUE) {
                    thresholds[example] = isNotLabelMax + 0.1;
                } else if (isNotLabelMax == -Double.MAX_VALUE) {
                    thresholds[example] = isLabelMin - 0.1;
                } else {
                    // center of a segment
                    thresholds[example] = (isLabelMin + isNotLabelMax) / 2;
                }
            } else {
                // when minimum is unique
                thresholds[example] = isLabelMin;
            }
        }

        Matrix modelMatrix = new Matrix(numExamples, numLabels + 1, 1.0);
        modelMatrix.setMatrix(0, numExamples - 1, 0, numLabels - 1, new Matrix(modelOutLabels));
        Matrix weights = modelMatrix.solve(new Matrix(thresholds, thresholds.length));
        double[][] weightsArray = weights.transpose().getArray();

        parameters = Arrays.copyOf(weightsArray[0], weightsArray[0].length);
    }

    /**
     * Returns parameters learned by the threshold function in last build.
     * Based on these parameters the functions is computing thresholds for
     * label confidences.<br>
     * Support for unit tests ...
     *
     * @return parameters
     */
    protected double[] getFunctionParameters() {
        return Arrays.copyOf(parameters, parameters.length);
    }
}