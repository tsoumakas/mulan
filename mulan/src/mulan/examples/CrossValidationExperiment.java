package mulan.examples;

/**
 *
 * @author greg
 */

import java.io.FileReader;

import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.RAKEL;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class CrossValidationExperiment {
    
    /**
     * Creates a new instance of this class
     */
    public CrossValidationExperiment() {
    }
    
	public static void main(String[] args) throws Exception
	{
            String path = "data/";
            String filename = "scene.arff";
            int numLabels = 6;

            FileReader frData = new FileReader(path + filename);
            Instances data = new Instances(frData);                
            
           
            
            Evaluator eval = new Evaluator(5);
            Evaluation results;

//            //* Binary Relevance Classifier
//            System.out.println("BR");
//            J48 brBaseClassifier = new J48();
//            BinaryRelevance br = new BinaryRelevance(brBaseClassifier,numLabels);
//            results = eval.crossValidate(br, data, 10);           
//            System.out.println(results.toString());
//            System.gc();
//            //*/

            //* Label Powerset Classifier
            System.out.println("LP");
            J48 lpBaseClassifier = new J48();
            LabelPowerset lp = new LabelPowerset(lpBaseClassifier, numLabels);
            results = eval.crossValidate(lp, data, 10);
            System.out.println(results.toString());
            System.gc();
            //*/
            
            //* RAKEL
            System.out.println("RAKEL");
            J48 rakelBaseClassifier = new J48();
            RAKEL rakel = new RAKEL(rakelBaseClassifier, numLabels, 10, 3);
            rakel.setParamSelectionViaCV(true);
            rakel.setParamSets(3, 2, numLabels-1, 1, 500, 0.1, 0.1, 9);
            results = eval.crossValidate(rakel, data, 10);
            System.out.println(results.toString());
            System.gc();                
            //*/            
            
            //* ML-kNN 
            System.out.println("ML-kNN");
            int numNeighbours = 10;
            MLkNN mlknn = new MLkNN(numLabels, numNeighbours, 1);
            results = eval.crossValidate(mlknn, data, 10);
            System.out.println(results.toString());
            System.gc();                
            //*/    
            
            //* BR-kNN 
            System.out.println("BR-kNN");
            numNeighbours = 10;
            BRkNN brknn = new BRkNN(numLabels, numNeighbours, 0);
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
