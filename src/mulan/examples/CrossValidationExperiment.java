package mulan.examples;

/**
 *
 * @author greg
 */

import java.io.FileReader;

import mulan.classifier.MultiLabelLearner;
import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.RAkEL;
import mulan.core.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

public class CrossValidationExperiment {

    /**
     * Creates a new instance of this class
     */
    public CrossValidationExperiment() {
    }

	public static void main(String[] args) throws Exception
	{
            String path = Utils.getOption("path", args);
            String filename = Utils.getOption("filename", args);
            int numLabels = Integer.parseInt(Utils.getOption("labels",args));

            MultiLabelInstances data = new MultiLabelInstances(path + filename, numLabels);	
            Evaluator eval = new Evaluator(5);
            Evaluation results;

            //* Binary Relevance Classifier
            System.out.println("BR");
            J48 brBaseClassifier = new J48();
            BinaryRelevance br = new BinaryRelevance(brBaseClassifier);
            results = eval.crossValidate(br, data, 10);
            System.out.println(results.toString());
            System.gc();
            //*/

            //* Label Powerset Classifier
            System.out.println("LP");
            J48 lpBaseClassifier = new J48();
            LabelPowerset lp = new LabelPowerset(lpBaseClassifier);
            results = eval.crossValidate(lp, data, 10);
            System.out.println(results.toString());
            System.gc();
            //*/

            //* RAKEL
            System.out.println("RAKEL");
            J48 rakelBaseClassifier = new J48();
            MultiLabelLearner lpBase = new LabelPowerset(new J48());
            RAkEL rakel = new RAkEL(lpBase);
            results = eval.crossValidate(rakel, data, 10);
            System.out.println(results.toString());
            System.gc();
            //*/

            //* ML-kNN
            System.out.println("ML-kNN");
            int numNeighbours = 10;
            MLkNN mlknn = new MLkNN(numNeighbours, 1);
            results = eval.crossValidate(mlknn, data, 10);
            System.out.println(results.toString());
            System.gc();
            //*/

            //* BR-kNN
            System.out.println("BR-kNN");
            numNeighbours = 10;
            BRkNN brknn = new BRkNN(numNeighbours, 0);
            brknn.setkSelectionViaCV(true);
            brknn.setCvMaxK(30);
            brknn.setDebug(true);
            brknn.build(data);
            //results = eval.crossValidateAll(brknn, data, 10);
            //System.out.println(results.toString());
            System.gc();
            //*/
        }
}
