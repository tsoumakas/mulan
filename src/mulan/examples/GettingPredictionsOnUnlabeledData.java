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
 *    GettingPredictionsOnUnlabeledData.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.examples;

import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Utils;

/**
 * This examples shows how you can retrieve the predictions of a model on
 * unlabeled data. Unlabeled multi-label datasets should have the same
 * structure as the training data. The actual values of the labels could be
 * either unspecified (set to symbol ?), or randomly set to 0/1.
 *
 * @author Grigorios Tsoumakas
 */
public class GettingPredictionsOnUnlabeledData {

    public static void main(String[] args) {

        try {
            String trainingDataFilename = Utils.getOption("training", args);
            String unlabeledDataFilename = Utils.getOption("unlabeled", args);
            String labelsFilename = Utils.getOption("labels", args);
            System.out.println("Loading the training data set...");
            MultiLabelInstances trainingData = new MultiLabelInstances(trainingDataFilename, labelsFilename);
            System.out.println("Loading the unlabeled data set...");
            MultiLabelInstances unlabeledData = new MultiLabelInstances(unlabeledDataFilename, labelsFilename);

            BinaryRelevance learner = new BinaryRelevance(new J48());

            System.out.println("Building the model...");
            learner.build(trainingData);
            int numInstances = unlabeledData.getDataSet().numInstances();
            for (int i = 0; i < numInstances; i++) {
                Instance instance = unlabeledData.getDataSet().instance(i);
                MultiLabelOutput output = learner.makePrediction(instance);
                if (output.hasBipartition()) {
                    String bipartion = Arrays.toString(output.getBipartition());
                    System.out.println("Predicted bipartion: " + bipartion);
                }
                if (output.hasRanking()) {
                    String ranking = Arrays.toString(output.getRanking());
                    System.out.println("Predicted ranking: " + ranking);
                }
                if (output.hasConfidences()) {
                    String confidences = Arrays.toString(output.getConfidences());
                    System.out.println("Predicted confidences: " + confidences);
                }
            }
        } catch (InvalidDataFormatException e) {
            System.err.println(e.getMessage());
        } catch (Exception ex) {
            Logger.getLogger(GettingPredictionsOnUnlabeledData.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}
