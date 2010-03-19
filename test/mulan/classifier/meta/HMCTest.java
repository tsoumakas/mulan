package mulan.classifier.meta;

import junit.framework.Assert;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.transformation.BinaryRelevance;
import mulan.data.MultiLabelInstances;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.SwapValues;

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


    @Override
    public void testBuildWith10ClassLabels() throws Exception {
        String trainDatasetPath = path + "hierarchical-train.arff";
        String testDatasetPath = path + "hierarchical-test.arff";
        String xmlLabelsDefFilePath = path + "hierarchical.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(
                trainDatasetPath, xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(
                testDatasetPath, xmlLabelsDefFilePath);
        String trainDatasetPath2 = path + "hierarchical10-train.arff";
        String testDatasetPath2 = path + "hierarchical10-test.arff";
        MultiLabelInstances trainDataSet2 = new MultiLabelInstances(
                trainDatasetPath2, xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet2 = new MultiLabelInstances(
                testDatasetPath2, xmlLabelsDefFilePath);


        MultiLabelLearner learner1 = getLearner();
        MultiLabelLearner learner2 = learner1.makeCopy();
        learner1.build(trainDataSet);
        learner2.build(trainDataSet2);
        for (int i = 0; i < testDataSet.getDataSet().numInstances(); i++) {
            MultiLabelOutput mlo1 = learner1.makePrediction(testDataSet.getDataSet().instance(i));
            MultiLabelOutput mlo2 = learner2.makePrediction(testDataSet2.getDataSet().instance(i));
            Assert.assertEquals(mlo1, mlo2);
        }
    }

}
