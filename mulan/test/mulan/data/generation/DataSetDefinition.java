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
package mulan.data.generation;

import java.util.ArrayList;
import java.util.List;

public class DataSetDefinition{
	
	private final List<Attribute> attributes;
	private int examplesCount = 100;
	private final String name;
	
	public DataSetDefinition(String dataSetName){
		name = dataSetName;
		attributes = new ArrayList<Attribute>();
	}
	
	public DataSetDefinition addAttribute(Attribute attribute){
		attributes.add(attribute);
		return this;
	}
	
	protected List<Attribute> getAttributes(){
		return attributes;
	}
	
	public DataSetDefinition setExamplesCount(int examplesCount){
		this.examplesCount = examplesCount;
		return this;
	}
	
	protected int getExamplesCount(){
		return examplesCount;
	}
	
	protected String getName(){
		return name;
	}
	
	public int getLabelsCount(){
		int numLabels = 0;
		for(Attribute attribute : attributes){
			if(attribute.isLabelAttribute()){
				numLabels++;
			}
		}
		return numLabels;
	}
	
}
