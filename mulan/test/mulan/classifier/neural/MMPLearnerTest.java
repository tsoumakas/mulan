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

/*
*    MMPLearnerTest.java
*    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
*
*/

package mulan.classifier.neural;

import junit.framework.Assert;
import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.MultiLabelLearnerTestBase;
import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;
import mulan.data.MultiLabelInstances;
import mulan.data.generation.DataSetBuilder;

import org.junit.Before;
import org.junit.Test;

import weka.core.Instance;


public class MMPLearnerTest extends MultiLabelLearnerTestBase {

	private MMPLearner learner;
	
	@Override
	protected MultiLabelLearnerBase getLearner() {
		return learner;
	}
	
	@Before
	public void setUp(){
		learner = new MMPLearner(LossMeasure.AveragePrecision, MMPUpdateRuleType.UniformUpdate);
	}

	@Test
	public void testTestDefaults(){
		Assert.assertEquals(true, learner.getConvertNominalToBinary());
		Assert.assertEquals(true, learner.getNormalizeAttributes());
		Assert.assertTrue(learner.isUpdatable());
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testConstructorWithNullLoss(){
		new MMPLearner(null, MMPUpdateRuleType.UniformUpdate);
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testConstructorWithNullUpdateRule(){
		new MMPLearner(LossMeasure.AveragePrecision, null);
	}
	
	@Test()
	public void testSetConvertNominalToBinary(){
		learner.setConvertNominalToBinary(false);
		Assert.assertFalse(learner.getConvertNominalToBinary());
	}
	
	@Test()
	public void testSetNormalizeAttributes(){
		learner.setNormalizeAttributes(false);
		Assert.assertFalse(learner.getNormalizeAttributes());
	}
			
	@Test(expected=InvalidDataException.class)
	public void testMakePrediction_WithInvalidData() throws Exception{
		MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
		learner.build(mlDataSet);
		
		Instance instance = new Instance(1);
		learner.makePrediction(instance);
	}
	
	@Test
	public void testMakePrediction() throws Exception{
		MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
		
		learner.build(mlDataSet);
		
		MultiLabelOutput prediction = learner.makePrediction(mlDataSet.getDataSet().instance(0));
		
		Assert.assertNotNull(prediction);
		Assert.assertNull(prediction.getBipartition());
		Assert.assertNull(prediction.getConfidences());
		Assert.assertNotNull(prediction.getRanking());
	}
	
	@Test
	public void testDifferentLossAndUpdateRules() throws Exception{
		MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
		
		MMPLearner learner;
		MultiLabelOutput prediction;
		
		learner = new MMPLearner(LossMeasure.AveragePrecision, MMPUpdateRuleType.UniformUpdate);
		learner.build(mlDataSet);
		prediction = learner.makePrediction(mlDataSet.getDataSet().instance(0));
		Assert.assertNotNull(prediction);
		
		learner = new MMPLearner(LossMeasure.ErrorSetSize, MMPUpdateRuleType.UniformUpdate);
		learner.build(mlDataSet);
		prediction = learner.makePrediction(mlDataSet.getDataSet().instance(0));
		Assert.assertNotNull(prediction);

		learner = new MMPLearner(LossMeasure.IsError, MMPUpdateRuleType.UniformUpdate);
		learner.build(mlDataSet);
		prediction = learner.makePrediction(mlDataSet.getDataSet().instance(0));
		Assert.assertNotNull(prediction);

		learner = new MMPLearner(LossMeasure.OneError, MMPUpdateRuleType.UniformUpdate);
		learner.build(mlDataSet);
		prediction = learner.makePrediction(mlDataSet.getDataSet().instance(0));
		Assert.assertNotNull(prediction);
		
		learner = new MMPLearner(LossMeasure.ErrorSetSize, MMPUpdateRuleType.RandomizedUpdate);
		learner.build(mlDataSet);
		prediction = learner.makePrediction(mlDataSet.getDataSet().instance(0));
		Assert.assertNotNull(prediction);

		learner = new MMPLearner(LossMeasure.IsError, MMPUpdateRuleType.MaxUpdate);
		learner.build(mlDataSet);
		prediction = learner.makePrediction(mlDataSet.getDataSet().instance(0));
		Assert.assertNotNull(prediction);

	}
}
