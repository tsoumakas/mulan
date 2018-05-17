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

import java.util.HashSet;
import java.util.Set;

import junit.framework.Assert;
import mulan.core.ArgumentNullException;

import org.junit.Before;
import org.junit.Test;

/**
 * Unit test routines for {@link LabelNodeImpl}.
 * 
 * @author Jozef Vilcek
 */
public class LabelNodeImplTest {

	private final String LABEL_NODE_NAME = "label node";
	private final String LEVEL_1_NODE_1 = "child_1";
	private final String LEVEL_1_NODE_2 = "child_2";
	private final String LEVEL_2_NODE_1 = "child_3";
	private final int ALL_CHILDREN_COUNT = 3;
	
	private LabelNodeImpl labelNode;
	
	@Before
	public void setUp(){
		labelNode = new LabelNodeImpl(LABEL_NODE_NAME);
		
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testConstructor_WithNullLabelName(){
		new LabelNodeImpl(null);
	}
	
	@Test
	public void testConstructor(){
		LabelNodeImpl node = new LabelNodeImpl(LABEL_NODE_NAME);
		Assert.assertEquals(0, node.getChildren().size());
		Assert.assertNull(node.getParent());
		Assert.assertEquals(LABEL_NODE_NAME, node.getName());
	}

	@Test
	public void testEquals(){
		Assert.assertFalse(labelNode.equals(null));
		Assert.assertFalse(labelNode.equals(new LabelNodeImpl("")));
		Assert.assertTrue(labelNode.equals(new LabelNodeImpl(LABEL_NODE_NAME)));
	}
	
	@Test
	public void testAddGetChildren(){
		Assert.assertEquals(0, labelNode.getChildren().size());
		Set<LabelNode> children = getLabelNodesForTest();
		for (LabelNode child : children) {
			Assert.assertTrue(labelNode.addChildNode(child));
		}
		Assert.assertEquals(children.size(), labelNode.getChildren().size());
		
		// check if every child has parent correctly set
		for (LabelNode item : children) {
			Assert.assertEquals(item.getParent(), labelNode);
		}
		
		// test same items are not added more than once
		for (LabelNode child : children) {
			Assert.assertFalse(labelNode.addChildNode(child));
		}
		Assert.assertEquals(children.size(), labelNode.getChildren().size());
	}
	
	@Test(expected=UnsupportedOperationException.class)
	public void testGetChildren_Unmodifiable(){
		labelNode.getChildren().add(new LabelNodeImpl(""));
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testAddChildren_WithNullNode(){
		labelNode.addChildNode(null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testAddChildren_WithSameLabelName(){
		labelNode.addChildNode(new LabelNodeImpl(LABEL_NODE_NAME));
	}
	
	@Test
	public void testGetChildrenLabels(){
		Assert.assertTrue(labelNode.getChildrenLabels().isEmpty());
		addLabelNodesForTest(labelNode);
		Assert.assertEquals(2, labelNode.getChildrenLabels().size());
	}
	
	@Test
	public void testGetDescendantLabels(){
		Assert.assertTrue(labelNode.getDescendantLabels().isEmpty());
		addLabelNodesForTest(labelNode);
		Assert.assertEquals(ALL_CHILDREN_COUNT, labelNode.getDescendantLabels().size());
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testRemoveChildNode_WithNull(){
		labelNode.removeChildNode(null);
	}
	
	@Test
	public void testRemoveChildNode(){
		LabelNode node = new LabelNodeImpl(LEVEL_1_NODE_1);
		labelNode.addChildNode(node);
		
		Assert.assertTrue(labelNode.hasChildren());
		Assert.assertTrue(labelNode.removeChildNode(node));
		Assert.assertFalse(labelNode.hasChildren());
		Assert.assertFalse(labelNode.removeChildNode(node));
	}
	
	
	private Set<LabelNode> addLabelNodesForTest(LabelNodeImpl labelNode){
		Set<LabelNode> nodes = getLabelNodesForTest();
		for (LabelNode child : nodes) {
			labelNode.addChildNode(child);
		}
		return nodes;
	}
	
	private Set<LabelNode> getLabelNodesForTest(){
		Set<LabelNode> nodes = new HashSet<LabelNode>();
		LabelNodeImpl node = new LabelNodeImpl(LEVEL_1_NODE_1);
		node.addChildNode(new LabelNodeImpl(LEVEL_2_NODE_1));
		nodes.add(node);
		nodes.add(new LabelNodeImpl(LEVEL_1_NODE_2));
		
		return nodes;
	}
	

}
