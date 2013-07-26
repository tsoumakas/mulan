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
package mulan.classifier.neural;

import java.util.Arrays;

import junit.framework.Assert;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;
import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import mulan.data.generation.Attribute;
import mulan.data.generation.DataSetBuilder;
import mulan.data.generation.DataSetDefinition;

import org.junit.Before;
import org.junit.Test;

import weka.core.DenseInstance;
import weka.core.Instance;

public class BPMLLTest extends MultiLabelLearnerTestBase {

    private static final double DEFAULT_LEARNING_RATE = 0.05;
    private static final double DEFAULT_WEIGTS_REGULARIZATION = 0.00001;
    private static final int DEFAULT_TRAIN_EPOCHS = 100;
    private static final boolean DEFAULT_NORMALIZATION = true;
    private BPMLL learner;

    @Override
    protected MultiLabelLearnerBase getLearner() {
        return learner;
    }

    @Before
    public void setUp() {
        learner = new BPMLL(10);
        learner.setTrainingEpochs(10);
        learner.setHiddenLayers(new int[]{});
    }

    @Test
    public void testTestDefaultParameters() {
        learner = new BPMLL();
        Assert.assertEquals(DEFAULT_LEARNING_RATE, learner.getLearningRate());
        Assert.assertEquals(DEFAULT_NORMALIZATION, learner.getNormalizeAttributes());
        Assert.assertEquals(DEFAULT_TRAIN_EPOCHS, learner.getTrainingEpochs());
        Assert.assertEquals(DEFAULT_WEIGTS_REGULARIZATION, learner.getWeightsDecayRegularization());

        // common tests
        Assert.assertFalse(learner.isUpdatable());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetHiddenLayers_WithInvalidLayer() {
        learner.setHiddenLayers(new int[]{2, 3, 0});
    }

    @Test
    public void testSetHiddenLayers_WithNull() {
        learner.setHiddenLayers(null);
        Assert.assertNull(learner.getHiddenLayers());
    }

    @Test
    public void testSetHiddenLayers_WithEmpty() {
        learner.setHiddenLayers(new int[]{});
    }

    @Test
    public void testSetHiddenLayers() {
        int[] expected = new int[]{1, 2, 3};
        learner.setHiddenLayers(expected);
        int[] actual = learner.getHiddenLayers();
        Assert.assertTrue(Arrays.equals(expected, actual));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetLearningRate_Invalid() {
        learner.setLearningRate(0);
    }

    @Test
    public void testSetLearningRate() {
        double learningRate = 1;
        learner.setLearningRate(learningRate);
        Assert.assertEquals(learningRate, learner.getLearningRate());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetWeightsDecayRegularization_InvalidLowerBound() {
        learner.setWeightsDecayRegularization(0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetWeightsDecayRegularization_InvalidUpperBound() {
        learner.setWeightsDecayRegularization(1.1);
    }

    @Test
    public void testSetWeightsDecayRegularization() {
        double value = 1;
        learner.setWeightsDecayRegularization(value);
        Assert.assertEquals(value, learner.getWeightsDecayRegularization());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetTrainigEpochs_WithInvalid() {
        learner.setTrainingEpochs(0);
    }

    @Test
    public void testSetTrainigEpochs() {
        int value = Integer.MAX_VALUE;
        learner.setTrainingEpochs(value);
        Assert.assertEquals(value, learner.getTrainingEpochs());
    }

    @Test
    public void testSetNormalizeAttributes() {
        learner.setNormalizeAttributes(false);
        Assert.assertFalse(learner.getNormalizeAttributes());
    }

    @Test
    public void testMakePrediction() throws Exception {
        MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
        learner.build(mlDataSet);

        MultiLabelOutput prediction = learner.makePrediction(mlDataSet.getDataSet().instance(0));

        Assert.assertNotNull(prediction);
        Assert.assertNotNull(prediction.getBipartition());
        Assert.assertNotNull(prediction.getConfidences());
        Assert.assertNotNull(prediction.getRanking());
    }

    @Test(expected = InvalidDataException.class)
    public void testMakePrediction_WithInvalidData() throws Exception {
        MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
        learner.build(mlDataSet);

        Instance instance = new DenseInstance(1);
        learner.makePrediction(instance);
    }

    @Test(expected = InvalidDataException.class)
    public void testBuild_WithNotSupportedDataSet() throws Exception {
        DataSetDefinition definition = new DataSetDefinition("NotSupportedDataSet");
        definition.addAttribute(Attribute.createNumericAttribute("feature_1"));
        definition.addAttribute(Attribute.createStringAttribute("feature_2"));
        definition.addAttribute(Attribute.createNominalAttribute("feature_3", new String[]{"n1", "n2", "n3", "n4"}));
        definition.addAttribute(Attribute.createLabelAttribute("label_1"));
        definition.addAttribute(Attribute.createLabelAttribute("label_2"));

        MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(definition);
        learner.build(mlDataSet);
    }
}
