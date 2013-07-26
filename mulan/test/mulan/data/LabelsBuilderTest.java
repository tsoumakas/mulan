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
import java.io.ByteArrayOutputStream;
import java.util.Set;

import junit.framework.Assert;

import org.junit.Test;

/**
 * Unit test routines for {@link LabelsBuilder}.
 * 
 * @author Jozef Vilcek
 */
public class LabelsBuilderTest {

	private final String LABELS_XML = 
			"<?xml version=\"1.0\" encoding=\"utf-8\"?>" +
			"<labels xmlns=\"http://mulan.sourceforge.net/labels\">" +
				"<label name=\"Class1\">" +
					"<label name=\"Class2\"></label>" +
					"<label name=\"Class3\"></label>" +
				"</label>" +
				"<label name=\"Class4\"></label>" +
			"</labels>";
	
	private final String HIERARCHY_NODE_NAME = "Class1";
	private final int NUM_HIERARCHY_NODE_CHILDREN = 2;
	private final int NUM_LABELS = 4;
	

	@Test(expected=LabelsBuilderException.class)
	public void testCreateLabels_WithInvalidXMLContent() throws LabelsBuilderException{
		ByteArrayInputStream inputStream = new ByteArrayInputStream("".getBytes());
		LabelsBuilder.createLabels(inputStream);
	}
	
	@Test
	public void testCreateLabels() throws LabelsBuilderException{
		ByteArrayInputStream inputStream = new ByteArrayInputStream(LABELS_XML.getBytes());
		LabelsMetaData metaData = LabelsBuilder.createLabels(inputStream);
		
		Assert.assertNotNull(metaData);
		Assert.assertTrue(metaData.isHierarchy());
		Assert.assertEquals(NUM_LABELS ,metaData.getNumLabels());
		
		LabelNode hierarchyNode = metaData.getLabelNode(HIERARCHY_NODE_NAME);
		Set<LabelNode> children = hierarchyNode.getChildren();
		Assert.assertEquals(NUM_HIERARCHY_NODE_CHILDREN, children.size());
	}
	
	@Test
	public void testDumpLabels() throws LabelsBuilderException{
		// create proper meta-data object (tested in other test-case)
		ByteArrayInputStream inputStream = new ByteArrayInputStream(LABELS_XML.getBytes());
		LabelsMetaData metaData = LabelsBuilder.createLabels(inputStream);
		
		// create an XML dump
		ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
		LabelsBuilder.dumpLabels(metaData, outputStream);
		
		// recreate the mata-data object from the dump
		metaData = LabelsBuilder.createLabels(new ByteArrayInputStream(outputStream.toByteArray()));
		
		// validate the meta-data object
		LabelNode hierarchyNode = metaData.getLabelNode(HIERARCHY_NODE_NAME);
		Set<LabelNode> children = hierarchyNode.getChildren();
		Assert.assertEquals(NUM_HIERARCHY_NODE_CHILDREN, children.size());
	}
}
