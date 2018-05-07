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

public class Attribute{
	private final AttributeType type;
	private final String name;
	private boolean isLabelAttribute = false;
	private double missingValuesProbability = 0;
	
	public static Attribute createNominalAttribute(String name, String[] values){
		return new NominalAttribute(name, values);
	}
	
	public static Attribute createLabelAttribute(String name){
		NominalAttribute attr = new NominalAttribute(name, new String[]{"0", "1"});
		attr.setIsLabelAttribute(true);
		return attr;
	}
	
	public static Attribute createNumericAttribute(String name, double min, double max){
		return new NumericAttribute(name, min, max);
	}
	
	public static Attribute createNumericAttribute(String name){
		return new NumericAttribute(name);
	}
	
	public static Attribute createStringAttribute(String name){
		return new Attribute(name, AttributeType.String);
	}
	
	public Attribute(String name, AttributeType type){
		this.name = name;
		this.type = type;
	}
	
	public AttributeType getType(){
		return type;
	}
	
	public String getName(){
		return name;
	}
	
	public Attribute setIsLabelAttribute(boolean value){
		isLabelAttribute = value;
		return this;
	}
	
	public boolean isLabelAttribute(){
		return isLabelAttribute;
	}
	
	public double getMissingValuesProbability(){
		return missingValuesProbability;
	}
	
	public Attribute setMissingValuesProbability(double probability){
		if(probability < 0 || probability > 1){
			throw new IllegalArgumentException("Probability must be form interval <0,1>.");
		}
		missingValuesProbability = probability;
		return this;
	}
	
	
	public enum AttributeType{
		Nominal,
		Numeric,
		String,
	}
}
