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
 *    LabelClustering.java
 */
package mulan.data;

/**
 * An interface for various label clustering algorithms.
 * 
 * @author Lena Chekina (lenat@bgu.ac.il)
 * @version 05.05.2011
 */
public interface LabelClustering {

	/**
	 * Returns a label set partitioning into clusters
	 * 
	 * @param trainingSet a set of training examples
	 * @return a label set partitioning
	 */
	public int[][] determineClusters(MultiLabelInstances trainingSet);

}