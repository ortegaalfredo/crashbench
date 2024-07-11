
/* https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=67f1aee6f45059fd6b0f5b0ecb2c97ad0451f6b3 
 Bug is at line 14:
-	return error;
+	return error < 0 ? error : 0;
*/




int iwch_l2t_send(struct t3cdev *tdev, struct sk_buff *skb, struct l2t_entry *l2e)
{
	int	error = 0;
	struct cxio_rdev *rdev;

	rdev = (struct cxio_rdev *)tdev->ulp;
	if (cxio_fatal_error(rdev)) {
		kfree_skb(skb);
		return -EIO;
	}
	error = l2t_send(tdev, skb, l2e);
	if (error < 0)
		kfree_skb(skb);
	return error;
}
