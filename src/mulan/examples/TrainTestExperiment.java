package mulan.examples;

import java.io.FileReader;
import mulan.classifier.RAKEL;
import mulan.evaluation.Evaluator;
import mulan.evaluation.IntegratedEvaluation;
import weka.classifiers.functions.SMO;
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
            IntegratedEvaluation results;

            //* RAKEL
            System.out.println("RAKEL");
            SMO rakelBaseClassifier = new SMO();
            RAKEL rakel = new RAKEL(rakelBaseClassifier, labels, 10, 3);
            //rakel.setParamSelectionViaCV(true);
            //rakel.setParamSets(3, 2, labels-1, 1, 500, 0.1, 0.1, 9);
            rakel.buildClassifier(train);
            results = eval.evaluateAll(rakel, test);
            System.out.println(results.toString());
            //*/
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

}
