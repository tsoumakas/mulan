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

import junit.framework.Assert;
import mulan.core.ArgumentNullException;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import weka.core.SerializedObject;

/**
 * Unit test routines for {@link LabelsMetaDataImpl}.
 * 
 * @author Jozef Vilcek
 */
public class LabelsMetaDataImplTest {

	private final String INVALID_LABEL_NAME = "invalid_label";
	private final String HIERARCHY_ROOT_NODE_NAME = "root";
	/** This value 'aaa' is specific and makes serialization fail if not handled explicitly in the implementation
	 *  The exact reason is unknown ... maybe the sequence of deserialization is influenced based on content dumped 
	 *  in to the serialization stream
	 * */
	private final String HIERARCHY_CHILD_1_NODE_NAME = "aaa"; 
	private final String HIERARCHY_CHILD_2_NODE_NAME = "child_2";
	private final int HIERACHY_NODES_COUNT = 3;
	private final int ROOT_NODES_COUNT = 2;
	private final String PLAIN_ROOT_NODE_NAME = "plain_root";
	
	private LabelsMetaDataImpl metaData;
	
	@Before
	public void setUp(){
		metaData = new LabelsMetaDataImpl();
		
		// add one root label node with hierarchy and one 'plain' node
		LabelNodeImpl root = new LabelNodeImpl(HIERARCHY_ROOT_NODE_NAME);
		root.addChildNode(new LabelNodeImpl(HIERARCHY_CHILD_1_NODE_NAME));
		root.addChildNode(new LabelNodeImpl(HIERARCHY_CHILD_2_NODE_NAME));
		
//		metaData.addRootNode(root);
		metaData.addRootNode(new LabelNodeImpl(PLAIN_ROOT_NODE_NAME));
		metaData.addRootNode(root);
	}
	
	@After
	public void tearDown(){
		metaData = null;
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testAddRootNode_WithNullNode(){
		metaData.addRootNode(null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testAddRootNode_WhichAlreadyExists(){
		metaData.addRootNode(new LabelNodeImpl(HIERARCHY_ROOT_NODE_NAME));
	}
	
	@Test
	public void testRemoveLabelNode_LabelDoesNotExists(){
		int numLabelNodes = metaData.getNumLabels();
		Assert.assertEquals(0, metaData.removeLabelNode(INVALID_LABEL_NAME));
		Assert.assertEquals(numLabelNodes, metaData.getNumLabels());
	}
	
	@Test
	public void testRemoveLabelNode_ChildNode(){
		int numLabelNodes = metaData.getNumLabels();
		
		Assert.assertEquals(1, metaData.removeLabelNode(HIERARCHY_CHILD_1_NODE_NAME));
		Assert.assertEquals(numLabelNodes - 1, metaData.getNumLabels());
		Assert.assertFalse(metaData.containsLabel(HIERARCHY_CHILD_1_NODE_NAME));
		Assert.assertEquals(ROOT_NODES_COUNT, metaData.getRootLabels().size());
	}
	
	@Test
	public void testRemoveLabelNode_RootNode(){
		int numLabelNodes = metaData.getNumLabels();
		
		Assert.assertEquals(HIERACHY_NODES_COUNT, 
							metaData.removeLabelNode(HIERARCHY_ROOT_NODE_NAME));
		Assert.assertEquals(numLabelNodes - HIERACHY_NODES_COUNT, metaData.getNumLabels());
		Assert.assertFalse(metaData.containsLabel(HIERARCHY_CHILD_1_NODE_NAME));
		Assert.assertFalse(metaData.containsLabel(HIERARCHY_CHILD_2_NODE_NAME));
		Assert.assertFalse(metaData.containsLabel(HIERARCHY_ROOT_NODE_NAME));
		Assert.assertEquals(ROOT_NODES_COUNT - 1, metaData.getRootLabels().size());
	}
	
	@Test
	public void testIsHierarchy(){
		Assert.assertTrue(metaData.isHierarchy());
		metaData.removeLabelNode(HIERARCHY_ROOT_NODE_NAME);
		Assert.assertFalse(metaData.isHierarchy());
	}
	
	@Test
	public void testClone(){
		LabelsMetaData clonedMetaData = metaData.clone();
		
		Assert.assertNotSame(metaData, clonedMetaData);
		Assert.assertEquals(metaData.isHierarchy(), clonedMetaData.isHierarchy());
		Assert.assertEquals(metaData.getNumLabels(), clonedMetaData.getNumLabels());
		Assert.assertEquals(metaData.getRootLabels().size(), clonedMetaData.getRootLabels().size());
		
		Assert.assertTrue(clonedMetaData.containsLabel(HIERARCHY_CHILD_1_NODE_NAME));
		Assert.assertTrue(clonedMetaData.containsLabel(HIERARCHY_CHILD_2_NODE_NAME));
		Assert.assertTrue(clonedMetaData.containsLabel(HIERARCHY_ROOT_NODE_NAME));
		Assert.assertTrue(clonedMetaData.containsLabel(PLAIN_ROOT_NODE_NAME));
	
	}
	
	@Test
	public void testSerialization() throws Exception {
		LabelsMetaData clonedMetaData = (LabelsMetaData) new SerializedObject(metaData).getObject();
		
		Assert.assertNotSame(metaData, clonedMetaData);
		Assert.assertEquals(metaData.isHierarchy(), clonedMetaData.isHierarchy());
		Assert.assertEquals(metaData.getNumLabels(), clonedMetaData.getNumLabels());
		Assert.assertEquals(metaData.getRootLabels().size(), clonedMetaData.getRootLabels().size());
		
		Assert.assertTrue(clonedMetaData.containsLabel(HIERARCHY_CHILD_1_NODE_NAME));
		Assert.assertTrue(clonedMetaData.containsLabel(HIERARCHY_CHILD_2_NODE_NAME));
		Assert.assertTrue(clonedMetaData.containsLabel(HIERARCHY_ROOT_NODE_NAME));
		Assert.assertTrue(clonedMetaData.containsLabel(PLAIN_ROOT_NODE_NAME));
	
	}
	

}
