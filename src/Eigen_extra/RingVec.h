#pragma once
#include "Eigen_extra.h"
#include <vector>
#include <utility>
#include <iterator>

namespace EigenExtra {

// 固定容量环形缓冲，存放标量序列，底层使用连续内存（Ve）
class RingVec {
public:
	RingVec() = default;
	explicit RingVec(int capacity) { resize(capacity); }

	void resize(int capacity);
	void clear();

	int capacity() const { return m_capacity; }
	int size() const { return m_size; }
	bool full() const { return m_size == m_capacity && m_capacity > 0; }
	bool empty() const { return m_size == 0; }

	// 头删尾添：写入 newValue，若满则返回被覆盖的旧值，否则返回 0.0
	double pushPop(double newValue);

	// 访问：从最旧元素起第 logicalIndex 个（0<=logicalIndex<size）
	double at(int logicalIndex) const;

	// 直接访问底层缓冲（物理顺序），仅供需要时使用
	const std::vector<double>& rawBuffer() const { return m_buf; }

	// 分段视图（零拷贝）：
	// 若未环回：firstSegment 覆盖全部数据，secondSegment 长度为 0；
	// 若已环回：两段依次覆盖“最旧 -> 最新”的逻辑顺序。
	std::pair<const double*, int> firstSegment() const;
	std::pair<const double*, int> secondSegment() const;

	// 转为 Eigen 向量（逻辑顺序：最旧 -> 最新）。会产生一次拷贝，适合参与 Ve 运算。
	Ve toVe() const;
	// 便捷隐式转换：允许在需要 Ve 的场景自动物化（注意有拷贝成本）。
	operator Ve() const { return toVe(); }
	// 用 Eigen 向量（逻辑顺序）整体重置内容；容量需一致。
	void assignFromVe(const Ve& sourceVector);

	// 迭代器支持（范围 for 循环）
	class const_iterator {
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = double;
		using difference_type = int;
		using pointer = const double*;
		using reference = const double&;

		const_iterator(const RingVec* ring, int logicalIndex) 
			: m_ring(ring), m_logicalIndex(logicalIndex) {}

		const double& operator*() const {
			// 直接访问底层缓冲，避免通过 at() 的按值返回
			const int physicalIndex = m_ring->logicalToPhysical(m_logicalIndex);
			return m_ring->m_buf[static_cast<size_t>(physicalIndex)];
		}

		const_iterator& operator++() {
			++m_logicalIndex;
			return *this;
		}

		const_iterator operator++(int) {
			const_iterator tmp = *this;
			++(*this);
			return tmp;
		}

		bool operator==(const const_iterator& other) const {
			return m_ring == other.m_ring && m_logicalIndex == other.m_logicalIndex;
		}

		bool operator!=(const const_iterator& other) const {
			return !(*this == other);
		}

	private:
		const RingVec* m_ring;
		int m_logicalIndex;
	};

	const_iterator begin() const {
		return const_iterator(this, 0);
	}

	const_iterator end() const {
		return const_iterator(this, m_size);
	}

private:
	// 将逻辑索引 i（相对最旧）映射到物理索引
	int logicalToPhysical(int logicalIndex) const;
	std::vector<double> m_buf; // 容量固定的连续存储
	int m_capacity{0}; // 容量（窗口长度）
	int m_size{0};     // 当前有效元素数
	int m_head{0};     // 下一次写入位置（物理索引）
};

} // namespace EigenExtra



