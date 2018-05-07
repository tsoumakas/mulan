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
package mulan.examples;

import java.util.logging.Level;
import java.util.logging.Logger;

import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.trees.J48;
import weka.core.SerializationHelper;
import weka.core.Utils;

/**
 * This example shows how you can store a learned model and load a stored model.
 * 
 * @author Grigorios Tsoumakas
 * @version 2012.02.06
 */
public class StoringAndLoadingModels {

    /**
     * Executes this example
     * 
     * @param args command-line arguments -train, -test -labels and -model, e.g. -train emotions-train.arff -test emotions-test.arff -labels emotions.xml -model model.dat
     */
    public static void main(String[] args) {
        try {
            String trainingDataFilename = Utils.getOption("train", args);
            String testingDataFilename = Utils.getOption("test", args);
            String labelsFilename = Utils.getOption("labels", args);
            System.out.println("Loading the training data set...");
            MultiLabelInstances trainingData = new MultiLabelInstances(trainingDataFilename, labelsFilename);
            System.out.println("Loading the testing data set...");
            MultiLabelInstances testingData = new MultiLabelInstances(testingDataFilename, labelsFilename);
            BinaryRelevance learner1 = new BinaryRelevance(new J48());

            String modelFilename = Utils.getOption("model", args);
            System.out.println("Building the model...");
            learner1.build(trainingData);

            System.out.println("Storing the model...");
            SerializationHelper.write(modelFilename, learner1);

            System.out.println("Loading the model...");
            BinaryRelevance learner2;
            learner2 = (BinaryRelevance) SerializationHelper.read(modelFilename);
            Evaluator evaluator = new Evaluator();
            Evaluation evaluation;
            evaluation = evaluator.evaluate(learner2, testingData, trainingData);
            System.out.println(evaluation);
        } catch (Exception ex) {
            Logger.getLogger(StoringAndLoadingModels.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}