
object closure{
  def mf(ini:Int)={
    var cs += ini
    val add=(a:Int)=>a+cs
  }
}

object closure{
  def main(args:Array[String])= {

    val t = closure.mf(0)
    println("res"+closure.add(2)) 
    println("res"+closure.add(3)) 
  }
}
