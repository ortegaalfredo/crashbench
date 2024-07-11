
/* https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=94f9cd81436c85d8c3a318ba92e236ede73752fc
 Bug is at line 29:
-		if (indev != NULL) {
+		if (indev && indev->ifa_list) {
*/




uunt nf_nat_redirect_ipv4(struct sk_buff *skb,
		     const struct nf_nat_ipv4_multi_range_compat *mr,
		     unsigned int hooknum)
{
	struct nf_conn *ct;
	enum ip_conntrack_info ctinfo;
	__be32 newdst;
	struct nf_nat_range newrange;

	NF_CT_ASSERT(hooknum == NF_INET_PRE_ROUTING ||
		     hooknum == NF_INET_LOCAL_OUT);

	ct = nf_ct_get(skb, &ctinfo);
	NF_CT_ASSERT(ct && (ctinfo == IP_CT_NEW || ctinfo == IP_CT_RELATED));

	/* Local packets: make them go to loopback */
	if (hooknum == NF_INET_LOCAL_OUT) {
		newdst = htonl(0x7F000001);
	} else {
		struct in_device *indev;
		struct in_ifaddr *ifa;

		newdst = 0;

		rcu_read_lock();
		indev = __in_dev_get_rcu(skb->dev);
		if (indev != NULL) {
			ifa = indev->ifa_list;
			newdst = ifa->ifa_local;
		}
		rcu_read_unlock();

		if (!newdst)
			return NF_DROP;
	}

	/* Transfer from original range. */
	memset(&newrange.min_addr, 0, sizeof(newrange.min_addr));
	memset(&newrange.max_addr, 0, sizeof(newrange.max_addr));
	newrange.flags	     = mr->range[0].flags | NF_NAT_RANGE_MAP_IPS;
	newrange.min_addr.ip = newdst;
	newrange.max_addr.ip = newdst;
	newrange.min_proto   = mr->range[0].min;
	newrange.max_proto   = mr->range[0].max;

	/* Hand modified range to generic setup. */
	return nf_nat_setup_info(ct, &newrange, NF_NAT_MANIP_DST);
}
