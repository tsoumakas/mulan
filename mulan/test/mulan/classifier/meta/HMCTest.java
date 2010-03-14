package mulan.classifier.meta;

import junit.framework.Assert;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;

public class HMCTest extends MultiLabelMetaLearnerTest {

    @Override
    public void setUp() throws Exception {
        learner = new HMC(new BinaryRelevance(new J48()));
    }

    @Override
    public void testMakeCopy() throws Exception {
        String trainDatasetPath = path + "hierarchical-train.arff";
        String testDatasetPath = path + "hierarchical-test.arff";
        String xmlLabelsDefFilePath = path + "hierarchical.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(
                trainDatasetPath, xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(
                testDatasetPath, xmlLabelsDefFilePath);

        learner.build(trainDataSet);

        MultiLabelLearnerBase copy = (MultiLabelLearnerBase) learner.makeCopy();

        for (int i = 0; i < testDataSet.getDataSet().numInstances(); i++) {
            MultiLabelOutput mlo1 = learner.makePrediction(testDataSet.getDataSet().instance(i));
            MultiLabelOutput mlo2 = copy.makePrediction(testDataSet.getDataSet().instance(i));
            Assert.assertEquals(mlo1, mlo2);
        }
    }
}
