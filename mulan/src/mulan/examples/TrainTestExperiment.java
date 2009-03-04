package mulan.examples;

import java.io.FileReader;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.MultiClassLearner;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.transformations.multiclass.*;
import mulan.transformations.multiclass.MultiClassTransformation;
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

            FileReader frTrain = new FileReader(path + filestem + "-train.arff");
            Instances train = new Instances(frTrain);
            FileReader frTest = new FileReader(path + filestem + "-test.arff");
            Instances test = new Instances(frTest);

            Evaluator eval = new Evaluator();
            Evaluation results;

            //* LP
            System.out.println("LP");
            J48 lpBaseClassifier = new J48();
            LabelPowerset lp = new LabelPowerset(lpBaseClassifier, labels);
            lp.build(train);
            results = eval.evaluate(lp, test);
            System.out.println(results.toString());
            //*/

            //* Multiclass Transformations
            System.out.println("Multiclass Transformations - Copy");
            J48 mcBaseClassifier = new J48();
            MultiClassTransformation copy = new Copy(labels);
            MultiClassLearner mc;
            mc = new MultiClassLearner(mcBaseClassifier, labels, copy);
            mc.build(train);
            results = eval.evaluate(mc, test);
            System.out.println(results.toString());

            //* BR
            System.out.println("BR");
            J48 brClassifier = new J48();
            BinaryRelevance br = new BinaryRelevance(brClassifier, labels);
            br.build(train);
            results = eval.evaluate(br, test);
            System.out.println(results.toString());
            //*/
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

}
