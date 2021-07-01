

object reg_op{
  def b2l(data:Int):List[Int]={
    List(data&0x00ff,(data>>>8)&0x00ff,(data>>>16)&0x00ff,(data>>>24)&0x00ff)
  }

  def l2b(data:List[Int]):Int={
    data.zipWithIndex.map(x=>x._1<<(8*x._2)).reduce(_|_)
  }

  def write_creg(addr:Int,data:Int):List[Int]={
    val SYNC = List(0x1b,0xdf,0x20,0x05)
    val frame = ((SYNC :+ addr)::: b2l(data))
    val calc =  frame.takeRight(5).reduce(_^_)
    (frame:+calc)
  }

  def read_creg(addr:Int,cnt:Int):List[Int]={
    val SYNC = List(0x1b,0xdf,0x10,0x02)
    val frame = (SYNC :+ addr :+ cnt)
    val calc =  frame.takeRight(2).reduce(_^_)
    (frame:+calc)
  }

  def read_sreg(addr:Int,cnt:Int):List[Int]={
    read_creg(addr|0x80,cnt)
  }

  def parse_frame(rxd_buf:List[Int]):Unit={
    val RSYNC = List(0x9b,0xdf)
    val sync_ind = rxd_buf.sliding(2).indexOf(RSYNC)
    if (sync_ind < 0 )
      return 

    val f_len = rxd_buf(sync_ind+3)
    val frame = rxd_buf.slice(sync_ind,sync_ind+5+f_len)
    val chk = frame.takeRight(f_len+1).reduce(_^_)
    val res = if (chk == 0){
      frame(2) match{
        case 0x10 => println("read resp:\nbase_addr:"+frame(2)+"\nreg_cnt:"+(f_len-1)/4+"\ndata:"+frame.drop(5).dropRight(1).grouped(4).toList.map(x=>l2b(x))+"\n")
        case 0x20 => println(s"write resp: ${if(0==frame(5)) "Ack" else "Nak"} \n")
        case _ => println("unrecongnizible frame"+"\n") 
      }
      rxd_buf.drop(sync_ind+5+f_len) 
    }
    else{
      println("checksum fail\n")
      rxd_buf.drop(sync_ind+2) 
    }
    parse_frame(res)
  }

  def main(args: Array[String]):Unit={
    val rd_resp = List(0x9b,0xdf,0x10,0x05,0x04,0x50,0xc3,0x00,0x00,0x97)
    val wr_resp = List(0x9b,0xdf,0x20,0x02,0x01,0x00,0x01)
    val rdn_resp = List(0x9b,0xdf,0x10,0x21,0x80,0x40,0x00,0x00,0x00,0x40,0x00,0x00,0x00,0x02,0x00,0x03,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01,0x00,0x00,0x00,0x24,0x01,0x00,0x00,0xa4)
    //向地址为19的配置寄存器写40
    println("\n>>>地址为19的配置寄存器写40<<<")
    println(write_creg(19,40).map(_.toHexString))
    //从地址为1的配置寄存器读取1个数据
    println("\n>>>从地址为1的配置寄存器读取1个数据<<<")
    println(read_creg(1,1).map(_.toHexString))
    //从地址为0的配置寄存器开始,连续读取8个数据(读取0-7寄存器内容)
    println("\n>>>从地址为0的配置寄存器开始,连续读取8个数据(读取0-7寄存器内容)<<<")
    println(read_creg(0,8).map(_.toHexString))
    //从地址为3的状态寄存器开始,读取8个数据
    println("\n>>>从地址为3的状态寄存器开始,读取8个数据<<<")
    println(read_sreg(13,8).map(_.toHexString))
    println("\n>>>rd resp parsing test<<<")
    parse_frame(rd_resp)
    println("\n>>>wr resp parsing test<<<")
    parse_frame(wr_resp)
    println("\n>>>rw resp parsing test<<<")
    parse_frame(wr_resp:::rd_resp)
    println("\n>>>rd multiple regs resp parsing test<<<")
    parse_frame(rdn_resp)
  }
}
