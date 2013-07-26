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
package mulan.data;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;

import junit.framework.Assert;
import mulan.core.ArgumentNullException;
import mulan.core.Util;

import org.junit.Before;
import org.junit.Test;

import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * Unit test routines for {@link LabelsBuilder}.
 * 
 * @author Jozef Vilcek
 */
public class MultiLabelInstancesTest {

	private final String LABELS_XML_STRING = 
		"<?xml version=\"1.0\" encoding=\"utf-8\"?>" +
		"<labels xmlns=\"http://mulan.sourceforge.net/labels\">" +
			"<label name=\"label1\"></label>" +
			"<label name=\"label2\">" +
				"<label name=\"label3\"></label>" +
			"</label>" +
		"</labels>";
	private final String DATA_SET_ARFF_STRING = 
		"@relation testDataSet" + Util.getNewLineSeparator() +
		"@attribute feature1 numeric" + Util.getNewLineSeparator() +
		"@attribute feature2 {1,2,3}" + Util.getNewLineSeparator() +
		"@attribute label1 {0,1}" + Util.getNewLineSeparator() +
		"@attribute label2 {0,1}" + Util.getNewLineSeparator() +
		"@attribute label3 {0,1}" + Util.getNewLineSeparator() +
		"@data" + Util.getNewLineSeparator() +
		"0.036299,2,0,0,0" + Util.getNewLineSeparator() +
		"0.161218,1,1,1,0" + Util.getNewLineSeparator() +
		"0.115987,3,0,1,1";
	
	private final int NUM_FEATURES = 2;
	private final int NUM_LABELS = 3;
	private MultiLabelInstances mlData;
	
	@Before
	public void setUp() throws InvalidDataFormatException, IOException{
		InputStream dataInputStream = new ByteArrayInputStream(DATA_SET_ARFF_STRING.getBytes());
		InputStream labelsInputStream = new ByteArrayInputStream(LABELS_XML_STRING.getBytes());
		mlData = new MultiLabelInstances(dataInputStream, labelsInputStream);
		dataInputStream.close();
		labelsInputStream.close();
	}
	
	@Test(expected=InvalidDataFormatException.class)
	public void testReintegrateModifiedDataSet_WithFailedValidationOnLabel() throws Exception{
		// remove label attributes (so only one remains)
		Remove remove = new Remove();
        remove.setAttributeIndicesArray(new int[]{NUM_FEATURES + NUM_LABELS - 1, NUM_FEATURES + NUM_LABELS - 2});
        remove.setInputFormat(mlData.getDataSet());
        Instances modifiedDataSet = Filter.useFilter(mlData.getDataSet(), remove);
        
        // try to re-integrate modification back into ML data set
        mlData.reintegrateModifiedDataSet(modifiedDataSet);
	}
	
	@Test
	public void testReintegrateModifiedDataSet() throws Exception{
		// remove label attribute
		Remove remove = new Remove();
        remove.setAttributeIndicesArray(new int[]{NUM_FEATURES + NUM_LABELS - 1});
        remove.setInputFormat(mlData.getDataSet());
        Instances modifiedDataSet = Filter.useFilter(mlData.getDataSet(), remove);
        
        // re-integrate modification back into ML data set
        MultiLabelInstances newMLData = mlData.reintegrateModifiedDataSet(modifiedDataSet);
        Assert.assertEquals(NUM_LABELS - 1, newMLData.getNumLabels());
	}
	
	@Test
	public void testClone(){
		MultiLabelInstances clone = mlData.clone();
		
		Assert.assertNotNull(clone);
		
		Assert.assertNotSame(mlData, clone);
		Assert.assertNotSame(mlData.getLabelsMetaData(), clone.getLabelsMetaData());
		Assert.assertNotSame(mlData.getDataSet(), clone.getDataSet());
		
		Assert.assertEquals(mlData.getNumLabels(), clone.getNumLabels());
		Assert.assertTrue(Arrays.equals(mlData.getLabelIndices(), clone.getLabelIndices()));
		Assert.assertTrue(Arrays.equals(mlData.getFeatureIndices(), clone.getFeatureIndices()));
	}
	
	@Test
	public void testConstructor() throws InvalidDataFormatException {
		Assert.assertNotNull(mlData);
		Assert.assertEquals(NUM_LABELS, mlData.getNumLabels());
		Assert.assertNotNull(mlData.getLabelsMetaData());
		Assert.assertNotNull(mlData.getDataSet());
		
		// check features
		Set<Attribute> features = mlData.getFeatureAttributes();
		int[] featureIndices = mlData.getFeatureIndices();
		Assert.assertNotNull(features);
		Assert.assertEquals(NUM_FEATURES, features.size());
		Assert.assertEquals(NUM_FEATURES, featureIndices.length);
		Assert.assertEquals(0, featureIndices[0]);
		Assert.assertEquals(1, featureIndices[1]);
		
		// check labels
		Set<Attribute> labels = mlData.getLabelAttributes();
		int[] labelIndices = mlData.getLabelIndices();
		Assert.assertNotNull(labels);
		Assert.assertEquals(NUM_LABELS, labels.size());
		Assert.assertEquals(NUM_LABELS, labelIndices.length);
		Assert.assertEquals(2, labelIndices[0]);
		Assert.assertEquals(3, labelIndices[1]);
		Assert.assertEquals(4, labelIndices[2]);
		
		// check labels order
		Map<String,Integer> mapping = mlData.getLabelsOrder();
		Assert.assertEquals((int)mapping.get("label1"), 0);
		Assert.assertEquals((int)mapping.get("label2"), 1);
		Assert.assertEquals((int)mapping.get("label3"), 2);
		
	}
	
	@Test(expected=InvalidDataFormatException.class)
	public void testConstructor_Stream_WithInvalidLabelAttributeFormat() throws InvalidDataFormatException{
		InputStream dataInputStream = new ByteArrayInputStream(DATA_SET_ARFF_STRING.getBytes());
		mlData = new MultiLabelInstances(dataInputStream, NUM_LABELS + 1);
	}
	
	@Test(expected=DataLoadException.class)
	public void testConstructor_WithInvalidLabelStream() throws InvalidDataFormatException{
		InputStream dataInputStream = new ByteArrayInputStream(DATA_SET_ARFF_STRING.getBytes());
		mlData = new MultiLabelInstances(dataInputStream, new ByteArrayInputStream(new byte[]{}));
	}
	
	@Test(expected=DataLoadException.class)
	public void testConstructor_WithInvalidDataSetStream() throws InvalidDataFormatException{
		InputStream labelsInputStream = new ByteArrayInputStream(LABELS_XML_STRING.getBytes());
		mlData = new MultiLabelInstances(new ByteArrayInputStream(new byte[]{}), labelsInputStream);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructor_WithInvalidLabelCount() throws InvalidDataFormatException{
		mlData = new MultiLabelInstances("", 1);
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testConstructor_WithNullDataFilePath() throws InvalidDataFormatException{
		mlData = new MultiLabelInstances((String)null, 2);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructor_NonExistingDataFilePath() throws InvalidDataFormatException{
		mlData = new MultiLabelInstances("", 2);
	}
	
	@Test(expected=DataLoadException.class)
	public void testConstructor_WithInvalidDataSetFile() throws InvalidDataFormatException{
		InputStream labelsInputStream = new ByteArrayInputStream(LABELS_XML_STRING.getBytes());
		mlData = new MultiLabelInstances(new ByteArrayInputStream(new byte[]{}), labelsInputStream);
	}
}
