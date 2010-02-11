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
 *    MultiLabelLearnerTestBase.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier;

import java.util.Arrays;

import junit.framework.Assert;
import mulan.core.ArgumentNullException;
import mulan.data.MultiLabelInstances;
import mulan.data.generation.Attribute;
import mulan.data.generation.DataSetBuilder;
import mulan.data.generation.DataSetDefinition;

import org.junit.Ignore;
import org.junit.Test;

@Ignore
public abstract class MultiLabelLearnerTestBase {

	/** General dataset which should be handled by all multi-label learners. Used to check learner in black-box mode */
	protected static final DataSetDefinition DATA_SET;

	/**
	 * Gets a learner instance which can be tested in each test case
	 * @return
	 */
	protected abstract MultiLabelLearnerBase getLearner();
		
	static {
		DATA_SET = new DataSetDefinition("GeneralDataSet");
		DATA_SET.addAttribute(Attribute.createNumericAttribute("feature_1"));
		DATA_SET.addAttribute(Attribute.createNumericAttribute("feature_2"));
		DATA_SET.addAttribute(Attribute.createNominalAttribute("feature_3", new String[]{"n1", "n2", "n3", "n4"}));
		DATA_SET.addAttribute(Attribute.createLabelAttribute("label_1"));
		DATA_SET.addAttribute(Attribute.createLabelAttribute("label_2"));
		DATA_SET.addAttribute(Attribute.createLabelAttribute("label_3"));
		DATA_SET.addAttribute(Attribute.createLabelAttribute("label_4"));
		DATA_SET.setExamplesCount(100);
	}
	
	
	@Test
	public void testGetTechnicalInformation(){
		Assert.assertNotNull(getLearner().getTechnicalInformation());
	}
	
	@Test
	public void testMakeCopy() throws Exception{
		MultiLabelLearnerBase learner = getLearner();
		MultiLabelLearnerBase copy = (MultiLabelLearnerBase) learner.makeCopy();
		
		// rough checks on obvious parameters ... interesting would be to compare learned models
		Assert.assertEquals(learner.numLabels, copy.numLabels);
		Assert.assertTrue(Arrays.equals(learner.featureIndices, copy.featureIndices));
		Assert.assertTrue(Arrays.equals(learner.labelIndices, copy.labelIndices));
	
		// build on copy
		MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
		copy.build(mlDataSet);
		
		// create a copy and test prediction - if build learner knowledge is persisted
		copy = (MultiLabelLearnerBase) copy.makeCopy();
		MultiLabelOutput prediction = copy.makePrediction(mlDataSet.getDataSet().firstInstance());
		// Note: precision is not tested - if original performs same as copy
		Assert.assertNotNull(prediction);
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testBuild_WithNullDataSet() throws Exception{
		getLearner().build(null);
	}
	
	@Test
	public void testBuild_WithMissingValues() throws Exception{
		DataSetDefinition definition = new DataSetDefinition("MissingValuesDataSet");
		definition.addAttribute(Attribute.createNumericAttribute("feature_1").setMissingValuesProbability(0.3));
		definition.addAttribute(Attribute.createNominalAttribute("feature_3", new String[]{"n1", "n2", "n3", "n4"}));
		definition.addAttribute(Attribute.createLabelAttribute("label_1"));
		definition.addAttribute(Attribute.createLabelAttribute("label_2"));
		
		MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(definition);
		getLearner().build(mlDataSet);
	}
	
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
	
	@Test(expected=ArgumentNullException.class)
	public void testMakePrediction_WithNullData() throws Exception {
		getLearner().makePrediction(null);
	}
	
	@Test(expected=ModelInitializationException.class)
	public void testMakePrediction_BeforeBuild() throws Exception {
		MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
		getLearner().makePrediction(mlDataSet.getDataSet().firstInstance());
	}
	
	@Test()
	public void testMakePrediction_Generic() throws Exception {
		MultiLabelInstances mlDataSet = DataSetBuilder.CreateDataSet(DATA_SET);
		MultiLabelLearnerBase learner = getLearner();
		
		learner.build(mlDataSet);
		MultiLabelOutput out = learner.makePrediction(mlDataSet.getDataSet().firstInstance());
		
		Assert.assertNotNull(out);
	}
}
