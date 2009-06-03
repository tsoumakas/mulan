package mulan.core;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import weka.core.AllJavadoc;
import weka.core.Utils;

/**
 * This class uses weka's Javadoc auto-generation classes to 
 * generate Javadoc comments and replace the content between certain comment tags.
 * 
 *  <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -classname &lt;classname&gt;
 *  The class to load. Fully qualified name (with package)</pre>
 * 
 * <pre> -dir &lt;dir&gt;
 *  The directory above the package hierarchy of the class.</pre>
 *  
 <!-- options-end -->
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 *
 */
public class MulanJavadoc {
	public static void main(String[] args) throws Exception {
		File file;
		BufferedWriter writer;

		String classname = Utils.getOption("classname", args);
		String dir = Utils.getOption("dir", args);
		String result = "";

		AllJavadoc jdoc = new AllJavadoc();
		jdoc.setClassname(classname);
		jdoc.setDir(dir);
		jdoc.setUseStars(false);
		
		try {
			result = jdoc.updateJavadoc();
			//System.out.println(result + "\n");
			file = new File(jdoc.getDir() + "/"
					+ jdoc.getClassname().replaceAll("\\.", "/") + ".java");
			if (!file.exists()) {
				System.out.println("File '" + file.getAbsolutePath()
						+ "' doesn't exist!");
				return;
			}
			writer = new BufferedWriter(new FileWriter(file));
			writer.write(result);
			writer.close();
			System.out.println(jdoc.getClassname() + "'s Javadoc successfully updated.");
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
