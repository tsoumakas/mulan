package mulan.examples;

import mulan.classifier.transformation.BinaryRelevance;
import mulan.classifier.transformation.CalibratedLabelRanking;
import mulan.classifier.transformation.LabelPowerset;
import mulan.classifier.transformation.RAkEL;
import mulan.core.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import weka.classifiers.trees.J48;
import weka.core.Utils;

/**
 *
 * @author greg
 */
public class TrainTestExperiment {

    public static void main(String[] args) {
        String[] methodsToCompare = {"RAkEL", "LP", "CLR", "BR"};

        try {
            String path = Utils.getOption("path", args);
            String filestem = Utils.getOption("filestem", args);
            MultiLabelInstances train = new MultiLabelInstances(path + filestem + "-train.arff", path + filestem + ".xml");
            MultiLabelInstances test = new MultiLabelInstances(path + filestem + "-test.arff", path + filestem + ".xml");
            
            Evaluator eval = new Evaluator();
            Evaluation results;

            for (int i=0; i<methodsToCompare.length; i++) {

                if (methodsToCompare[i].equals("BR")) {
                    System.out.println(methodsToCompare[i]);
                    J48 brClassifier = new J48();
                    BinaryRelevance br = new BinaryRelevance(brClassifier);
                    br.build(train);
                    results = eval.evaluate(br, test);
                    System.out.println(results.toString());
                }

                if (methodsToCompare[i].equals("CLR")) {
                    System.out.println(methodsToCompare[i]);
                    J48 clrClassifier = new J48();
                    CalibratedLabelRanking clr = new CalibratedLabelRanking(clrClassifier);
                    clr.setDebug(true);
                    clr.build(train);
                    results = eval.evaluate(clr, test);
                    System.out.println(results.toString());
                }

                if (methodsToCompare[i].equals("LP")) {
                    System.out.println(methodsToCompare[i]);
                    J48 lpBaseClassifier = new J48();
                    LabelPowerset lp = new LabelPowerset(lpBaseClassifier);
                    lp.setDebug(true);
                    lp.build(train);
                    results = eval.evaluate(lp, test);
                    System.out.println(results.toString());
                }

                if (methodsToCompare[i].equals("RAkEL")) {
                    System.out.println(methodsToCompare[i]);
                    LabelPowerset lp = new LabelPowerset(new J48());
                    RAkEL rakel = new RAkEL(lp);
                    rakel.setDebug(true);
                    rakel.build(train);
                    results = eval.evaluate(rakel, test);
                    System.out.println(results.toString());
                }
            }
        }
        catch (Exception e) {
            e.printStackTrace();
        }
    }

}
