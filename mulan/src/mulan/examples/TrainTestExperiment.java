package mulan.examples;

import java.io.FileReader;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.MultiClassLearner;
import mulan.core.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.transformations.multiclass.*;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.Utils;

/**
 *
 * @author greg
 */
public class TrainTestExperiment {

    public static void main(String[] args) {
        long before, after;
        double paramTime, buildTime, testTime;
        int numExperiments = 10;

        try {
            String path = Utils.getOption("path", args);
            String filestem = Utils.getOption("filestem", args);
            int labels = Integer.parseInt(Utils.getOption("labels",args));

            MultiLabelInstances train = new MultiLabelInstances(path + filestem + "-train.arff", labels);
            MultiLabelInstances test = new MultiLabelInstances(path + filestem + "-test.arff", labels);
            
            Evaluator eval = new Evaluator();
            Evaluation results;

            //* LP
            System.out.println("LP");
            J48 lpBaseClassifier = new J48();
            LabelPowerset lp = new LabelPowerset(lpBaseClassifier);
            lp.build(train);
            results = eval.evaluate(lp, test.getDataSet());
            System.out.println(results.toString());
            //*/

            //* Multiclass Transformations
            System.out.println("Multiclass Transformations - Copy");
            J48 mcBaseClassifier = new J48();
            MultiClassTransformation copy = new Copy(labels);
            MultiClassLearner mc;
            mc = new MultiClassLearner(mcBaseClassifier, copy);
            mc.build(train);
            results = eval.evaluate(mc, test.getDataSet());
            System.out.println(results.toString());

            //* BR
            System.out.println("BR");
            J48 brClassifier = new J48();
            BinaryRelevance br = new BinaryRelevance(brClassifier);
            br.build(train);
            results = eval.evaluate(br, test.getDataSet());
            System.out.println(results.toString());
            //*/
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

}
