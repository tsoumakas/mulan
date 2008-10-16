package mulan.examples;

/**
 *
 * @author greg
 */

import java.io.FileReader;

import mulan.classifier.BinaryRelevanceClassifier;
import mulan.classifier.LabelPowerset;
import mulan.classifier.RAKEL;
import mulan.classifier.lazy.MLkNN;
import mulan.evaluation.Evaluator;
import mulan.evaluation.IntegratedCrossvalidation;
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
            String path = "d:/work/datasets/multilabel/scene/";
            String filename = "scene.arff";
            int numLabels = 6;

            FileReader frData = new FileReader(path + filename);
            Instances data = new Instances(frData);                
            
           
            
            Evaluator eval = new Evaluator(5);
            IntegratedCrossvalidation results;

            //* Binary Relevance Classifier
            System.out.println("BR");
            BinaryRelevanceClassifier br = new BinaryRelevanceClassifier();
            J48 brBaseClassifier = new J48();
            br.setBaseClassifier(brBaseClassifier);
            br.setNumLabels(numLabels);
            results = eval.crossValidateAll(br, data, 10);           
            System.out.println(results.toString());
            System.gc();
            //*/

            //* Label Powerset Classifier
            System.out.println("LP");
            J48 lpBaseClassifier = new J48();
            LabelPowerset lp = new LabelPowerset(lpBaseClassifier, numLabels);
            results = eval.crossValidateAll(lp, data, 10);
            System.out.println(results.toString());
            System.gc();
            //*/
            
            //* RAKEL
            System.out.println("RAKEL");
            RAKEL rakel = new RAKEL(numLabels, 10, 3);
            J48 rakelBaseClassifier = new J48();
            rakel.setBaseClassifier(rakelBaseClassifier);
            rakel.setParamSelectionViaCV(true);
            rakel.setParamSets(3, 2, numLabels-1, 1, 500, 0.1, 0.1, 9);
            results = eval.crossValidateAll(rakel, data, 10);
            System.out.println(results.toString());
            System.gc();                
            //*/            
            
            //* ML-kNN 
            System.out.println("ML-kNN");
            int numNeighbours = 10;
            MLkNN mlknn = new MLkNN(numLabels, numNeighbours, 1);
            results = eval.crossValidateAll(mlknn, data, 10);
            System.out.println(results.toString());
            System.gc();                
            //*/        
        }    
}
