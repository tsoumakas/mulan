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
package mulan.classifier;

import junit.framework.Assert;
import mulan.core.ArgumentNullException;
import mulan.data.MultiLabelInstances;
import mulan.data.generation.Attribute;
import mulan.data.generation.DataSetBuilder;
import mulan.data.generation.DataSetDefinition;

import org.junit.Ignore;
import org.junit.Test;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Reorder;
import weka.filters.unsupervised.instance.SparseToNonSparse;

/**
 *
 * @author Grigorios Tsoumakas
 * @version 2013.07.05
 */
@Ignore
public abstract class MultiLabelLearnerTestBase {

    /**
     * Path to the data set used in tests. This dataset should be handled by all
     * multi-label learners. Used to check learners in black-box mode
     */
    protected String path = "./data/testData/";
    /**
     * Artificial dataset used in tests
     */
    protected static final DataSetDefinition DATA_SET;

    /**
     * Gets a learner instance which can be tested in each test case
     *
     * @return
     */
    protected abstract MultiLabelLearnerBase getLearner();

    static {
        DATA_SET = new DataSetDefinition("GeneralDataSet");
        DATA_SET.addAttribute(Attribute.createNumericAttribute("feature_1"));
        DATA_SET.addAttribute(Attribute.createNumericAttribute("feature_2"));
        DATA_SET.addAttribute(Attribute.createNominalAttribute("feature_3", 
                new String[]{"n1", "n2", "n3", "n4"}));
        DATA_SET.addAttribute(Attribute.createLabelAttribute("label_1"));
        DATA_SET.addAttribute(Attribute.createLabelAttribute("label_2"));
        DATA_SET.addAttribute(Attribute.createLabelAttribute("label_3"));
        DATA_SET.addAttribute(Attribute.createLabelAttribute("label_4"));
        DATA_SET.setExamplesCount(100);
    }

    /**
     * Tests if the learner implements the getTechnicalInformation method
     */
    @Test
    public void testGetTechnicalInformation() {
        Assert.assertNotNull(getLearner().getTechnicalInformation());
    }

    /**
     * Tests if the learner returned by the makeCopy method produces the same
     * output as the original learner
     *
     * @throws Exception
     */
    @Test
    public void testMakeCopy() throws Exception {
        MultiLabelLearnerBase learner = getLearner();

        String trainDatasetPath = path + "emotions-train.arff";
        String testDatasetPath = path + "emotions-test.arff";
        String xmlLabelsDefFilePath = path + "emotions.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(trainDatasetPath,
                xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(testDatasetPath,
                xmlLabelsDefFilePath);

        learner.build(trainDataSet);

        MultiLabelLearnerBase copy = (MultiLabelLearnerBase) learner.makeCopy();

        for (int i = 0; i < testDataSet.getDataSet().numInstances(); i++) {
            MultiLabelOutput mlo1 = learner.makePrediction(testDataSet.getDataSet().instance(i));
            MultiLabelOutput mlo2 = copy.makePrediction(testDataSet.getDataSet().instance(i));
            Assert.assertEquals(mlo1, mlo2);
        }
    }

    /**
     * Tests if the learner throws ArgumentNullException when build is called
     * with a null argument
     *
     * @throws Exception
     */
    @Test(expected = ArgumentNullException.class)
    public void testBuild_WithNullDataSet() throws Exception {
        getLearner().build(null);
    }

    /**
     * Tests if the learner can be built when the dataset has missing values in
     * the features or in the labels
     *
     * @throws Exception
     */
    @Test
    public void testBuild_WithMissingValues() throws Exception {
        DataSetDefinition definition = new DataSetDefinition("MissingValuesDataSet");
        definition.addAttribute(Attribute.createNumericAttribute("feature_1")
                .setMissingValuesProbability(0.3));
        definition.addAttribute(Attribute.createNominalAttribute("feature_2", 
                new String[]{"n1", "n2", "n3", "n4"}));
        definition.addAttribute(Attribute.createLabelAttribute("label_1")
                .setMissingValuesProbability(0.3));
        definition.addAttribute(Attribute.createLabelAttribute("label_2"));
        definition.addAttribute(Attribute.createLabelAttribute("label_3")
                .setMissingValuesProbability(0.5));
        definition.addAttribute(Attribute.createLabelAttribute("label_4"));

        MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(definition);
        getLearner().build(mlDataSet);
    }
    /**
     * Tests if the learner produces the same output when the label attributes
     * are defined as {1,0} instead of {0,1}
     *
     * @throws Exception
     */
    @Test
    public void testBuildWith10ClassLabels() throws Exception {
        String trainDatasetPath = path + "emotions-train.arff";
        String testDatasetPath = path + "emotions-test.arff";
        String xmlLabelsDefFilePath = path + "emotions.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(trainDatasetPath,
                xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(testDatasetPath,
                xmlLabelsDefFilePath);
        String trainDatasetPath2 = path + "emotions10-train.arff";
        String testDatasetPath2 = path + "emotions10-test.arff";
        MultiLabelInstances trainDataSet2 = new MultiLabelInstances(trainDatasetPath2,
                xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet2 = new MultiLabelInstances(testDatasetPath2,
                xmlLabelsDefFilePath);

        MultiLabelLearner learner1 = getLearner();
        MultiLabelLearner learner2 = learner1.makeCopy();
        learner1.build(trainDataSet);
        learner2.build(trainDataSet2);
        for (int i = 0; i < testDataSet.getDataSet().numInstances(); i++) {
            MultiLabelOutput mlo1 = learner1.makePrediction(testDataSet.getDataSet().instance(i));
            MultiLabelOutput mlo2 = learner2.makePrediction(testDataSet2.getDataSet().instance(i));
            Assert.assertEquals(mlo1, mlo2);
        }

        /*
         * Evaluator eval = new Evaluator(); Evaluation results;
         * 
         * MultiLabelLearnerBase learner = getLearner();
         * learner.build(trainDataSet); results = eval.evaluate(learner,
         * testDataSet); String firstRunResults = results.toString();
         * learner.build(trainDataSet2); results = eval.evaluate(learner,
         * testDataSet2); String secondRunResults = results.toString();
         * //System.out.println(firstRunResults);
         * //System.out.println(secondRunResults);
         * 
         * Assert.assertTrue(firstRunResults.equals(secondRunResults));
         */

    }

    /**
     * Tests if a learner always produces the same output when built with the
     * same data. This is to make sure that all learners handle randomization
     * issues properly (using seeds).
     *
     * @throws Exception
     */
    @Test
    public void testSameInputSameOutput() throws Exception {
        String trainDatasetPath = path + "emotions-train.arff";
        String testDatasetPath = path + "emotions-test.arff";
        String xmlLabelsDefFilePath = path + "emotions.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(trainDatasetPath,
                xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(testDatasetPath,
                xmlLabelsDefFilePath);

        MultiLabelLearner learner1 = getLearner();
        MultiLabelLearner learner2 = learner1.makeCopy();
        learner1.build(trainDataSet);
        learner2.build(trainDataSet);
        for (int i = 0; i < testDataSet.getDataSet().numInstances(); i++) {
            MultiLabelOutput mlo1 = learner1.makePrediction(testDataSet.getDataSet().instance(i));
            MultiLabelOutput mlo2 = learner2.makePrediction(testDataSet.getDataSet().instance(i));
            Assert.assertEquals(mlo1, mlo2);
        }
    }

    /**
     * Tests if the learner produces the same output when we re-order the
     * attributes (features and labels) in a dataset
     *
     * @throws Exception
     */
    @Test
    public void testBuild_WithDifferentOrder() throws Exception {
        String trainDatasetPath = path + "emotions-train.arff";
        String testDatasetPath = path + "emotions-test.arff";
        String xmlLabelsDefFilePath = path + "emotions.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(trainDatasetPath,
                xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(testDatasetPath,
                xmlLabelsDefFilePath);
        Instances originalTrainData = new Instances(trainDataSet.getDataSet());
        Instances originalTestData = new Instances(testDataSet.getDataSet());
        // tranform the dataset
        Reorder reorder = new Reorder();
        /*
         * Note: The re-ordering retains the relative order of labels. The
         * reason for this is, because we are comparing the outputs of the
         * learners, which are assembled in the order of label appearance in the
         * arff dataset. Alternatively we could compare the overall results, but
         * see the following note for a problem in this case.
         */
        reorder.setAttributeIndices("1-10,73,11-20,74,21-30,75,31-40,76,41-50,77,51-72,78");
        reorder.setInputFormat(originalTrainData);// inform filter about dataset
        // **AFTER** setting options
        Instances transformedTrainData = Filter.useFilter(originalTrainData, reorder);// apply
        // filter

        Instances transformedTestData = Filter.useFilter(originalTestData, reorder);// apply
        // filter

        MultiLabelInstances transformedTrainDataSet = new MultiLabelInstances(transformedTrainData,
                xmlLabelsDefFilePath);
        MultiLabelInstances transformedTestDataSet = new MultiLabelInstances(transformedTestData,
                xmlLabelsDefFilePath);

        MultiLabelLearner learner1 = getLearner();
        MultiLabelLearner learner2 = learner1.makeCopy();
        learner1.build(trainDataSet);
        learner2.build(transformedTrainDataSet);
        for (int i = 0; i < testDataSet.getDataSet().numInstances(); i++) {
            MultiLabelOutput mlo1 = learner1.makePrediction(testDataSet.getDataSet().instance(i));
            MultiLabelOutput mlo2 = learner2.makePrediction(transformedTestDataSet.getDataSet()
                    .instance(i));
            Assert.assertEquals(mlo1, mlo2);
        }

        /*
         * Note: it may be more meaningful to use the following type of test for
         * arbitrary reorderings of labels, but the problem lies in the
         * calculation of ranking measures, in the typical case where the
         * ranking is obtained from scores, solving the ties through
         * weka.core.StableSort. Different orderings of the labels will produce
         * small variations in ranking measures
         * 
         * Evaluator eval = new Evaluator(); Evaluation results;
         * 
         * results = eval.evaluate(learner1,testDataSet); String OriginalResults
         * = results.toString(); results = eval.evaluate(learner2,
         * transformedTestDataSet); String resultsAfterTransformation =
         * results.toString(); System.out.println(this.getClass().toString());
         * System.out.println(OriginalResults);
         * System.out.println(resultsAfterTransformation);
         * Assert.assertTrue(OriginalResults
         * .equals(resultsAfterTransformation));
         */
    }

    /**
     * Tests if the learner produces the same output when the dataset is in non
     * sparse format.
     *
     * @throws Exception
     */
    @Test
    public void testBuildWithNonSparse() throws Exception {
        String trainDatasetPath = path + "sparseDataSet-train.arff";
        String testDatasetPath = path + "sparseDataSet-test.arff";
        String xmlLabelsDefFilePath = path + "sparseDataSet.xml";
        MultiLabelInstances trainDataSet = new MultiLabelInstances(trainDatasetPath,
                xmlLabelsDefFilePath);
        MultiLabelInstances testDataSet = new MultiLabelInstances(testDatasetPath,
                xmlLabelsDefFilePath);
        Instances originalTrainData = new Instances(trainDataSet.getDataSet());
        Instances originalTestData = new Instances(testDataSet.getDataSet());

        // tranform the datasets
        SparseToNonSparse nsp = new SparseToNonSparse();
        nsp.setInputFormat(originalTrainData);// inform filter about dataset
        // **AFTER** setting options
        Instances transformedTrainData = Filter.useFilter(originalTrainData, nsp);// apply
        // filter
        MultiLabelInstances transformedTrainDataSet = new MultiLabelInstances(transformedTrainData,
                xmlLabelsDefFilePath);

        Instances transformedTestData = Filter.useFilter(originalTestData, nsp);// apply
        // filter
        MultiLabelInstances transformedTestDataSet = new MultiLabelInstances(transformedTestData,
                xmlLabelsDefFilePath);

        MultiLabelLearner learner1 = getLearner();
        MultiLabelLearner learner2 = learner1.makeCopy();
        learner1.build(trainDataSet);
        learner2.build(transformedTrainDataSet);
        for (int i = 0; i < testDataSet.getDataSet().numInstances(); i++) {
            MultiLabelOutput mlo1 = learner1.makePrediction(testDataSet.getDataSet().instance(i));
            MultiLabelOutput mlo2 = learner2.makePrediction(transformedTestDataSet.getDataSet()
                    .instance(i));
            Assert.assertEquals(mlo1, mlo2);
        }
    }

    /**
     * Tests if the build method correctly initializes the learner
     *
     * @throws Exception
     */
    @Test
    public void testBuild() throws Exception {
        MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
        MultiLabelLearnerBase learner = getLearner();

        Assert.assertNull(learner.labelIndices);
        Assert.assertNull(learner.featureIndices);
        Assert.assertEquals(0, learner.numLabels);

        learner.build(mlDataSet);

        Assert.assertNotNull(learner.labelIndices);
        Assert.assertNotNull(learner.featureIndices);
        Assert.assertEquals(DATA_SET.getLabelsCount(), learner.numLabels);

    }

    /**
     * Tests if the correct exception is thrown when the learner's
     * makePrediction method is called with a null argument
     *
     * @throws Exception
     */
    @Test(expected = ArgumentNullException.class)
    public void testMakePrediction_WithNullData() throws Exception {
        getLearner().makePrediction(null);
    }

    /**
     * Tests if the correct exception is thrown when makePrediction is called
     * before build
     *
     * @throws Exception
     */
    @Test(expected = ModelInitializationException.class)
    public void testMakePrediction_BeforeBuild() throws Exception {
        MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
        getLearner().makePrediction(mlDataSet.getDataSet().firstInstance());
    }

    /**
     * Generic makePrediction test
     *
     * @throws Exception
     */
    @Test()
    public void testMakePrediction_Generic() throws Exception {
        MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
        MultiLabelLearnerBase learner = getLearner();

        learner.build(mlDataSet);
        MultiLabelOutput out = learner.makePrediction(mlDataSet.getDataSet().firstInstance());

        Assert.assertNotNull(out);
    }
}
